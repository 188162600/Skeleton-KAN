import copy
import unittest
import torch
from structured_kan.model import StructuredKANBuilder,KAN
from structured_kan.model.StructuredKANBuilder import _AffineRoute
from structured_kan.model.shared_affine_spec import shared_affine_private_phi_spec,shared_affine_layout,initialize_private_from_shared_
from structured_kan.model.initialization import perturb_trainable_


def fixture():
    return dict(builder='example',synthetic_operator_tree=dict(operator='+',raw_arity=1,
        children=[dict(operator='*',raw_arity=2,children=[])]),
        raw_incidence_rule=dict(port_capacity_by_node={'__shared_pool__':dict(core_capacity_by_node={'root':1,'root.0':2},reserve_capacity=1)},
        role_incidence_prior={'mode':'independent-prefix'},activation_prior_by_node={'root':{'initial_active_lanes':1},'root.0':{'initial_active_lanes':2}}),
        edge_route_rule={'state_route_prior':{}})


class SharedAffinePrivatePhiTest(unittest.TestCase):
    def test_private_routes_paired_seed_initialization_and_vmap(self):
        shared_spec,shared_meta=shared_affine_private_phi_spec(fixture(),2)
        spec,meta=shared_affine_private_phi_spec(fixture(),2,share_affine=False)
        self.assertEqual(meta['shared_affine_roles'],0)
        self.assertEqual(meta['affine_route_modules'],5)
        self.assertEqual(meta['expected_parameters'],50)
        x=torch.randn(19,2,dtype=torch.float64);models=[];digests=[]
        for seed in range(421,431):
            shared=StructuredKANBuilder(2).build(shared_spec,train_inputs=x)
            expected_hash=perturb_trainable_(shared,seed)
            private=StructuredKANBuilder(2).build(spec,train_inputs=x)
            digest,source_hash=initialize_private_from_shared_(private,shared_spec,x,seed)
            self.assertEqual(source_hash,expected_hash);digests.append(digest)
            self.assertEqual(private.parameter_count(),meta['expected_parameters'])
            affines=[m for m in private.modules() if isinstance(m,_AffineRoute)]
            self.assertEqual(len({m.weights.data_ptr() for m in affines}),5)
            self.assertEqual(len([m for m in private.modules() if isinstance(m,KAN)]),7)
            torch.testing.assert_close(private(x),shared(x),rtol=0,atol=0)
            for key,value in private.state_dict().items():torch.testing.assert_close(value,shared.state_dict()[key],rtol=0,atol=0)
            models.append(private)
        self.assertEqual(len(set(digests)),10)
        params,buffers=torch.func.stack_module_state(models);template=copy.deepcopy(models[0])
        def loss(p,b):return torch.func.functional_call(template,(p,b),(x,)).square().mean()
        grads,values=torch.vmap(torch.func.grad_and_value(loss))(params,buffers)
        for i,m in enumerate(models):
            val=m(x).square().mean();val.backward();torch.testing.assert_close(val,values[i])
            for name,p in m.named_parameters():torch.testing.assert_close(p.grad,grads[name][i])
        parent=models[0].branches[0];first=parent.branches[0];other=parent.branches[-1].branches[0]
        before=other.weights.detach().clone()
        with torch.no_grad():first.weights.add_(.1)
        torch.testing.assert_close(before,other.weights,rtol=0,atol=0)

    def test_roles_private_maps_and_neutral_init(self):
        spec,meta=shared_affine_private_phi_spec(fixture(),2)
        model=StructuredKANBuilder(2).build(spec)
        self.assertEqual(meta['raw_routes'],5);self.assertEqual(meta['shared_affine_roles'],3)
        self.assertEqual(meta['scalar_edges'],7);self.assertEqual(model.parameter_count(),44)
        raw=[m for m in model.modules() if isinstance(m,_AffineRoute)]
        phis=[m for m in model.modules() if isinstance(m,KAN)]
        self.assertEqual(len(raw),3);self.assertEqual(len(phis),7)
        self.assertEqual(len({id(m.coefficients) for m in phis}),7)
        x=torch.tensor([[2.,3.],[1.,4.]],dtype=torch.float64)
        torch.testing.assert_close(model(x),x[:,:1]+x[:,:1]*x[:,1:2])
        root=model.branches[0];product=root.branches[-1]
        self.assertIs(root.branches[0],product.branches[0])
        self.assertIsNot(root.phi[0],product.phi[0])

    def test_reserve_ids_survive_dimension_change(self):
        b=fixture()
        for d in (1,2,5):
            layout=shared_affine_layout(b,d)
            self.assertEqual(len(layout['roles']),min(d,2)+1)
            self.assertEqual([e['key'] for e in layout['edges'] if ':reserve:' in e['key']],['root.0:reserve:0','root:reserve:0'])
        b['private_phi_G_by_route']={e['key']:(15 if ':reserve:' in e['key'] else 3) for e in shared_affine_layout(b,5)['edges']}
        for d in (1,2,5):
            spec,meta=shared_affine_private_phi_spec(b,d)
            model=StructuredKANBuilder(d).build(spec)
            self.assertEqual(model.parameter_count(),meta['expected_parameters'])

    def test_shared_gradients_equal_sum_of_clones(self):
        spec,meta=shared_affine_private_phi_spec(fixture(),2)
        shared=StructuredKANBuilder(2).build(spec)
        separate=copy.deepcopy(spec)
        def untie(n):
            n.pop('share',None)
            for c in n.get('children',[]):untie(c)
        untie(separate);private=StructuredKANBuilder(2).build(separate)
        self.assertEqual(private.parameter_count()-shared.parameter_count(),6)
        x=torch.randn(17,2,dtype=torch.float64)
        shared(x).square().mean().backward();private(x).square().mean().backward()
        a=shared.branches[0];b=private.branches[0]
        torch.testing.assert_close(a.branches[0].weights.grad,
            b.branches[0].weights.grad+b.branches[-1].branches[0].weights.grad)

    def test_vmap_with_shared_module_matches_seed_loop(self):
        spec,_=shared_affine_private_phi_spec(fixture(),2)
        x=torch.randn(19,2,dtype=torch.float64)
        models=[StructuredKANBuilder(2).build(spec,train_inputs=x) for _ in range(10)]
        digests=[perturb_trainable_(m,421+i) for i,m in enumerate(models)]
        self.assertEqual(len(set(digests)),10)
        params,buffers=torch.func.stack_module_state(models);template=copy.deepcopy(models[0])
        canonical={id(v):n for n,v in template.named_parameters()}
        canonical.update({id(v):n for n,v in template.named_buffers()})
        pa={};ba={}
        for mn,module in template.named_modules():
            prefix=mn+'.' if mn else ''
            for n,v in module.named_parameters(recurse=False):pa[prefix+n]=canonical[id(v)]
            for n,v in module.named_buffers(recurse=False):ba[prefix+n]=canonical[id(v)]
        def loss(p,b):
            return torch.func.functional_call(template,({a:p[n] for a,n in pa.items()},{a:b[n] for a,n in ba.items()}),(x,),tie_weights=False).square().mean()
        grads,values=torch.vmap(torch.func.grad_and_value(loss))(params,buffers)
        for i,m in enumerate(models):
            val=m(x).square().mean();val.backward();torch.testing.assert_close(values[i],val)
            for name,p in m.named_parameters():torch.testing.assert_close(grads[name][i],p.grad)


if __name__=='__main__':unittest.main()
