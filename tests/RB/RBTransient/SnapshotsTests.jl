using LinearAlgebra
using Test
using Gridap
using Gridap.Helpers

using Mabla.FEM
using Mabla.FEM.IndexMaps
using Mabla.FEM.ParamDataStructures
using Mabla.FEM.ParamFESpaces
using Mabla.FEM.ParamSteady
using Mabla.FEM.ParamODEs

using Mabla.RB
using Mabla.RB.RBSteady
using Mabla.RB.RBTransient

ns = 6
nt = 1
np = 2
pranges = fill([0,1],3)
tdomain = 0:1
ptspace = TransientParamSpace(pranges,tdomain)
r = realization(ptspace,nparams=np)
i = TrivialIndexMap(LinearIndices((ns,)))

vv = [collect((ip-1)*ns+1:ip*ns) for ip = 1:np*nt]
s = Snapshots(ParamArray(vv),i,r)
@test norm(dropdims(s;dims=2) - stack(vv)) ≈ 0
s[12] = 13
@check s[12] == 13
s[12] = 12

vals = get_values(s)

s′ = Snapshots([ParamArray(vv)],i,r)
@test norm(s - s′) ≈ 0
s[12] = 13
@check s[12] == 13
s[12] = 12

prange = 1:1
s′′ = select_snapshots(s,prange,trange=1:1)
@check norm(s′′ - s[:,1,1]) ≈ 0

r′ = realization(TransientParamSpace(pranges,0:3),nparams=np)
s′′′ = Snapshots([ParamArray(vv),2 .* ParamArray(vv),3 .* ParamArray(vv)],i,r′)
vals = get_values(s′′′)

entries = select_snapshots_entries(s′′′,1:5,1:2)

s′′′′ = flatten_snapshots(s′′′)
s′′′′[5,5] = 27

ϕ = rand(6,3)
compress(s′′′′,ϕ)
