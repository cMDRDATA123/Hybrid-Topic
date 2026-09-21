import numpy as np
from experiments.propagation_ablation import seed_matrix, query_assign

def test_propagation_can_reach_query_without_changing_gate():
    # Query sees only second reference. No propagation leaves it without seed evidence.
    refs=np.eye(2);query=np.array([[0.,1.]])
    y=seed_matrix(np.array([[.9,.1],[.2,.8]]),1)
    assert np.array_equal(y,np.array([[.9,0],[0,.8]]))
    empty_second=np.array([[1.,0.],[0.,0.]])
    assert query_assign(query,refs,empty_second,.5,.7)[0][0]==-1
    assert query_assign(query,refs,np.array([[1.,0.],[.2,0.]]),.5,.7)[0][0]==0
    assert query_assign(np.array([[-1.,0.]]),refs,y,.5,.7)[0][0]==-1
