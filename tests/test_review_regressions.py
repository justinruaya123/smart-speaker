import numpy as np
import pytest


def test_calibration_uses_runtime_acceptance():
    from scripts.calibrate_semifinal import acceptance_mask
    from vcm.acceptance import decide
    command={'intent':'POWER_ON','entity_id':'light.a'}
    rows=[{'candidate':command,'score':-1.,'margin':m} for m in (0.,.2)]
    rows += [{'candidate':None,'score':-1.,'margin':.2},
             {'candidate':command,'score':float('nan'),'margin':.2},
             {'candidate':command,'score':-1.,'margin':float('inf')}]
    policy={'enabled':True,'binding':{},'min_score':-2.,'min_margin':0.}
    expected=[decide(r['candidate'],r['score'],r['margin'],policy,{},endpoint=True)['accepted'] for r in rows]
    assert acceptance_mask(rows,-2.,0.).tolist()==expected==[False,True,False,False,False]


def test_features_require_frozen_bytes_and_frontend(tmp_path):
    from vcm.data import feature_receipt,load_verified_features
    path=tmp_path/'f.npy'; np.save(path,np.ones((40,3),dtype='float32'))
    row={'features':str(path),**feature_receipt(path)}
    assert load_verified_features(row).shape==(40,3)
    with pytest.raises(ValueError,match='provenance'): load_verified_features({'features':str(path)})
    with pytest.raises(ValueError,match='frontend'): load_verified_features({**row,'frontend_spec':'wrong'})
    np.save(path,np.zeros((40,3),dtype='float32'))
    with pytest.raises(ValueError,match='hash'): load_verified_features(row)
