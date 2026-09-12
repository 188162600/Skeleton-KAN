import copy
import unittest
import torch
from structured_kan.model import StructuredKANBuilder,KAN
from structured_kan.model.StructuredKANBuilder import _AffineRoute
from structured_kan.model.shared_affine_spec import private_affine_shared_phi_spec,shared_affine_layout,shared_affine_private_phi_spec
from structured_kan.model.initialization import perturb_trainable_
from structured_kan.tests.test_shared_affine_private_phi import fixture
from structured_kan.scripts.benchmark import equation_indices,DOMAINS


class PrivateAffineSharedPhiTests(unittest.TestCase):
    def test_aliases_and_parameters(self):
        spec,meta=private_affine_shared_phi_spec(fixture(),2)
        model=StructuredKANBuilder(2).build(spec)
        root=model.branches[0];child=root.branches[-1]
        self.assertIsNot(root.branches[0],child.branches[0])
        self.assertIs(root.phi[0],child.phi[0])
        self.assertIs(root.phi[0].a,child.phi[0].a)
        self.assertIs(root.phi[0].coefficients,child.phi[0].coefficients)
        # Neutral sum reserve=0 and product reserve=1 must NOT share.
        self.assertIsNot(root.phi[1],child.phi[2])
        self.assertEqual(meta['phi_reused_occurrences'],1)
        self.assertEqual(len([m for m in model.modules() if isinstance(m,KAN)]),6)
        self.assertEqual(len([m for m in model.modules() if isinstance(m,_AffineRoute)]),5)
        self.assertEqual(model.parameter_count(),45)
        self.assertEqual(model.parameter_count(),meta['expected_parameters'])
        before=child.branches[0].weights.detach().clone()
        with torch.no_grad():root.branches[0].weights.add_(.1)
        torch.testing.assert_close(child.branches[0].weights,before,rtol=0,atol=0)

    def test_incompatible_G_not_tied(self):
        b=fixture()
        b['private_phi_G_by_route']={e['key']:3 for e in shared_affine_layout(b,2)['edges']}
        b['private_phi_G_by_route']['root:raw:0']=9
        spec,meta=private_affine_shared_phi_spec(b,2)
        model=StructuredKANBuilder(2).build(spec)
        self.assertEqual(meta['phi_reused_occurrences'],0)
        self.assertEqual(model.parameter_count(),56)

    def test_forward_and_tied_gradient_equal_private_clone_sum(self):
        spec,_=private_affine_shared_phi_spec(fixture(),2)
        untied=copy.deepcopy(spec)
        def untie(node):
            for p in node.get('phi',[]):p.pop('share',None)
            for c in node.get('children',[]):untie(c)
        untie(untied)
        x=torch.randn(23,2,dtype=torch.float64)
        shared=StructuredKANBuilder(2).build(spec,train_inputs=x)
        private=StructuredKANBuilder(2).build(untied,train_inputs=x)
        perturb_trainable_(shared,421);private.load_state_dict(shared.state_dict())
        torch.testing.assert_close(shared(x),private(x),rtol=0,atol=0)
        shared(x).square().mean().backward();private(x).square().mean().backward()
        a=shared.branches[0];b=private.branches[0]
        for name,p in a.phi[0].named_parameters():
            expected=dict(b.phi[0].named_parameters())[name].grad+dict(b.branches[-1].phi[0].named_parameters())[name].grad
            torch.testing.assert_close(p.grad,expected)

    def test_batched_gradient_matches_ten_seed_loop(self):
        spec,_=private_affine_shared_phi_spec(fixture(),2)
        x=torch.randn(10,19,2,dtype=torch.float64);models=[];digests=[]
        for i in range(10):
            m=StructuredKANBuilder(2).build(spec,train_inputs=x[i])
            digests.append(perturb_trainable_(m,421+i));models.append(m)
        self.assertEqual(len(set(digests)),10)
        params,buffers=torch.func.stack_module_state(models);template=copy.deepcopy(models[0])
        canonical={id(v):n for n,v in template.named_parameters()}
        canonical.update({id(v):n for n,v in template.named_buffers()})
        pa={};ba={}
        for mn,module in template.named_modules():
            prefix=mn+'.' if mn else ''
            for n,v in module.named_parameters(recurse=False):pa[prefix+n]=canonical[id(v)]
            for n,v in module.named_buffers(recurse=False):ba[prefix+n]=canonical[id(v)]
        def loss(p,b,xx):
            return torch.func.functional_call(template,({a:p[n] for a,n in pa.items()},{a:b[n] for a,n in ba.items()}),(xx,),tie_weights=False).square().mean()
        grads,values=torch.vmap(torch.func.grad_and_value(loss))(params,buffers,x)
        for i,m in enumerate(models):
            val=m(x[i]).square().mean();val.backward();torch.testing.assert_close(values[i],val)
            for name,p in m.named_parameters():torch.testing.assert_close(grads[name][i],p.grad)

    def test_equation_shards_are_disjoint_complete_and_domain_balanced(self):
        rows=[dict(source_corpus=d) for d in DOMAINS for _ in range(20)]
        shards=[equation_indices(dict(equation_partition=dict(index=i,count=2,policy='within-domain-round-robin')),rows) for i in range(2)]
        self.assertEqual(set(shards[0])&set(shards[1]),set())
        self.assertEqual(set(shards[0])|set(shards[1]),set(range(80)))
        for shard in shards:
            self.assertEqual(len(shard),40)
            for domain in DOMAINS:self.assertEqual(sum(rows[i]['source_corpus']==domain for i in shard),10)
        self.assertEqual(equation_indices({},rows),list(range(80)))


if __name__=='__main__':unittest.main()
