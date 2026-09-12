"""Independent-seed compact L-BFGS with coalesced full-data evaluations.

The controller is relocated from the established seed-batched implementation:
50 outer calls by default, at most20 inner iterations, history100, independent
strong-Wolfe searches and the original1e-32 stopping tolerances. No model or
dataset imports live in the optimization core.
"""
from __future__ import annotations

import concurrent.futures
import dataclasses
import queue
import threading
from typing import Any

import torch

from .line_search import _strong_wolfe_coalesced


@dataclasses.dataclass
class Request:
    seed_index: int
    x: torch.Tensor
    t: float
    d: torch.Tensor
    ready: threading.Event = dataclasses.field(default_factory=threading.Event)
    response: tuple | None = None


@dataclasses.dataclass(frozen=True)
class LBFGSResult:
    parameters: torch.Tensor
    loss: torch.Tensor
    inner_iterations: tuple[int, ...]
    function_evaluations: tuple[int, ...]


def minimize(initial, value_and_gradient, value, *, outer_steps=50, progress=None):
    """Optimize rows independently; callbacks evaluate a permanent seed batch.

    value_and_gradient(rows, seed_indices) -> (gradient, per_row_loss).
    value(rows) evaluates all original seed objectives in order. Callbacks
    must use full datasets and return independent row objectives, not a mean
    across seeds. Grid buffers can differ by seed but model shapes must match.
    """
    if initial.ndim != 2 or initial.dtype != torch.float64 or not initial.numel():
        raise ValueError("initial parameters must be a nonempty float64 (seeds, parameters) matrix")
    if type(outer_steps) is not int or outer_steps < 1:
        raise ValueError("outer_steps must be a positive integer")
    flat = initial.detach().clone()
    batch_size, parameter_count = flat.shape
    device = flat.device
    all_indices = torch.arange(batch_size,device=device)

    def grad_value(rows, indices):
        gradients, losses = value_and_gradient(rows,indices)
        return gradients.detach().clone(), losses.detach().clone()

    # The unchanged controller body is mechanically relocated below.

    request_queue: queue.Queue[tuple[str, int, Request | None]] = queue.Queue()

    def evaluate_requests(requests: list[Request]) -> None:
        real_count = len(requests)
        padded = list(requests)
        while len(padded) < batch_size:
            padded.append(requests[0])
        indices = torch.tensor(
            [request.seed_index for request in padded],
            device=device,
            dtype=torch.long,
        )
        trial = torch.stack(tuple(
            request.x + request.t * request.d for request in padded
        ))
        gradients, losses = grad_value(trial, indices)
        directions = torch.stack(tuple(request.d for request in padded))
        directional = torch.sum(gradients * directions, dim=1)
        gradient_max = gradients.abs().amax(dim=1)
        scalars = torch.stack((losses, directional, gradient_max), dim=1).tolist()
        for position, request in enumerate(requests[:real_count]):
            loss_item, directional_item, maximum_item = scalars[position]
            request.response = (
                loss_item,
                gradients[position],
                directional_item,
                maximum_item,
            )
            request.ready.set()

    def line_search(
        x: torch.Tensor,
        initial_t: torch.Tensor,
        directions: torch.Tensor,
        losses: torch.Tensor,
        gradients: torch.Tensor,
        seed_indices: list[int],
    ) -> dict[int, tuple[float, torch.Tensor, float, int]]:
        initial_metrics = torch.stack((
            torch.sum(gradients * directions, dim=1),
            gradients.abs().amax(dim=1),
            directions.abs().amax(dim=1),
        ), dim=1).tolist()
        results: dict[int, Any] = {}

        def worker(seed_index: int) -> None:
            def objective(x_row: torch.Tensor, t: float, d_row: torch.Tensor):
                request = Request(seed_index, x_row, float(t), d_row)
                request_queue.put(("request", seed_index, request))
                request.ready.wait()
                if request.response is None:
                    raise RuntimeError("missing batched line-search response")
                return request.response

            gtd, gradient_max, direction_norm = initial_metrics[seed_index]
            try:
                results[seed_index] = _strong_wolfe_coalesced(
                    objective,
                    x[seed_index],
                    float(initial_t[seed_index]),
                    directions[seed_index],
                    float(losses[seed_index]),
                    gradients[seed_index],
                    gtd,
                    gradient_max,
                    direction_norm,
                    tolerance_change=1.0e-32,
                )
            finally:
                request_queue.put(("done", seed_index, None))

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(seed_indices)
        ) as executor:
            futures = [executor.submit(worker, index) for index in seed_indices]
            active = len(seed_indices)
            while active:
                messages = [request_queue.get() for _ in range(active)]
                requests = [
                    message[2] for message in messages
                    if message[0] == "request" and message[2] is not None
                ]
                active -= sum(message[0] == "done" for message in messages)
                if requests:
                    evaluate_requests(requests)
            for future in futures:
                future.result()

        return {
            index: (
                float(results[index][0]),
                results[index][1],
                float(results[index][2]),
                int(results[index][3]),
            )
            for index in seed_indices
        }

    best_loss = value(flat)
    best_flat = flat.clone()
    history_size = 100
    directions_history = torch.zeros(
        batch_size, parameter_count, history_size,
        dtype=torch.float64, device=device)
    steps_history = torch.zeros_like(directions_history)
    rho_history = torch.zeros(
        batch_size, history_size, dtype=torch.float64, device=device)
    history_counts = [0] * batch_size
    state_iterations = [0] * batch_size
    previous_gradient = torch.zeros_like(flat)
    direction = torch.zeros_like(flat)
    accepted_t = torch.zeros(batch_size, dtype=torch.float64, device=device)
    h_diag = torch.ones(batch_size, dtype=torch.float64, device=device)
    total_iterations = [0] * batch_size
    total_evaluations = [0] * batch_size

    for _outer in range(outer_steps):
        gradient, loss = grad_value(flat, all_indices)
        gradient = gradient.detach()
        loss = loss.detach()
        current_evaluations = [1] * batch_size
        total_evaluations = [value + 1 for value in total_evaluations]
        active = set(range(batch_size))
        for _inner in range(20):
            if not active:
                break
            active_list = sorted(active)
            for index in active_list:
                state_iterations[index] += 1
                total_iterations[index] += 1
                if state_iterations[index] == 1:
                    direction[index] = -gradient[index]
                    continue
                y_delta = gradient[index] - previous_gradient[index]
                s_delta = direction[index] * accepted_t[index]
                ys = float(torch.sum(y_delta * s_delta))
                if ys <= 1.0e-32:
                    continue
                count = history_counts[index]
                if count == history_size:
                    directions_history[index, :, :-1] = (
                        directions_history[index, :, 1:].clone())
                    steps_history[index, :, :-1] = (
                        steps_history[index, :, 1:].clone())
                    rho_history[index, :-1] = rho_history[index, 1:].clone()
                    slot = history_size - 1
                else:
                    slot = count
                    history_counts[index] += 1
                directions_history[index, :, slot] = y_delta
                steps_history[index, :, slot] = s_delta
                rho_history[index, slot] = 1.0 / ys
                h_diag[index] = ys / torch.sum(y_delta * y_delta)

            maximum_history = max(history_counts[index] for index in active_list)
            if maximum_history:
                directions_used = directions_history[:, :, :maximum_history]
                steps_used = steps_history[:, :, :maximum_history]
                rho = rho_history[:, :maximum_history]
                sty = torch.bmm(
                    steps_used.transpose(1, 2), directions_used)
                identity = torch.eye(
                    maximum_history, dtype=torch.float64, device=device
                ).expand(batch_size, -1, -1)
                q0 = -gradient
                upper = identity + torch.triu(sty, diagonal=1) * rho[:, :, None]
                alpha = torch.linalg.solve_triangular(
                    upper,
                    (rho * torch.bmm(
                        steps_used.transpose(1, 2), q0.unsqueeze(2)
                    ).squeeze(2)).unsqueeze(2),
                    upper=True,
                ).squeeze(2)
                q = q0 - torch.bmm(
                    directions_used, alpha.unsqueeze(2)).squeeze(2)
                initial = q * h_diag[:, None]
                lower = (
                    identity
                    + torch.tril(sty.transpose(1, 2), diagonal=-1)
                    * rho[:, :, None]
                )
                correction = torch.linalg.solve_triangular(
                    lower,
                    (alpha - rho * torch.bmm(
                        directions_used.transpose(1, 2), initial.unsqueeze(2)
                    ).squeeze(2)).unsqueeze(2),
                    upper=False,
                ).squeeze(2)
                proposed_direction = initial + torch.bmm(
                    steps_used, correction.unsqueeze(2)).squeeze(2)
                direction[active_list] = proposed_direction[active_list]

            previous_gradient[active_list] = gradient[active_list]
            previous_loss = loss.clone()
            initial_t = torch.ones(
                batch_size, device=device, dtype=torch.float64)
            first_indices = [
                index for index in active_list if state_iterations[index] == 1
            ]
            if first_indices:
                initial_t[first_indices] = torch.minimum(
                    torch.ones(
                        len(first_indices), device=device, dtype=torch.float64),
                    1.0 / gradient[first_indices].abs().sum(dim=1),
                )

            gtd = torch.sum(gradient * direction, dim=1).tolist()
            for index in list(active):
                if gtd[index] > -1.0e-32:
                    active.remove(index)
            search_indices = sorted(active)
            if search_indices:
                results = line_search(
                    flat, initial_t, direction, loss, gradient, search_indices)
                for index, result in results.items():
                    loss_value, gradient_value, step_value, evaluations = result
                    loss[index] = loss_value
                    gradient[index] = gradient_value
                    accepted_t[index] = step_value
                    flat[index] = flat[index] + step_value * direction[index]
                    current_evaluations[index] += evaluations
                    total_evaluations[index] += evaluations

            if _inner != 19:
                stops = torch.stack((
                    gradient.abs().amax(dim=1),
                    (direction * accepted_t[:, None]).abs().amax(dim=1),
                    (loss - previous_loss).abs(),
                ), dim=1)
                stop_values = stops.tolist()
                for index in list(active):
                    if (
                        min(stop_values[index]) <= 1.0e-32
                        or current_evaluations[index] >= 25
                    ):
                        active.remove(index)

        current = value(flat)
        improved = current < best_loss
        best_flat = torch.where(improved[:, None], flat, best_flat)
        best_loss = torch.where(improved, current, best_loss)
        if progress is not None:
            progress(_outer+1,best_loss.detach(),tuple(total_iterations),tuple(total_evaluations))

    return LBFGSResult(best_flat, best_loss, tuple(total_iterations), tuple(total_evaluations))
