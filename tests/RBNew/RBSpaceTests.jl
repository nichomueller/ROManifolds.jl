using LinearAlgebra
using Plots
using Test
using Gridap
using Gridap.Helpers
using Mabla.FEM
using Mabla.RB

ns = 100
nt = 10
np = 5
pranges = fill([0,1],3)
tdomain = 0:1:10
ptspace = TransientParamSpace(pranges,tdomain)
r = realization(ptspace,nparams=np)

v = [rand(ns) for i = 1:np*nt]
a = ParamArray(v)
s = Snapshots(a,r)
Us,Ss,Vs = svd(s)
s2 = RB.change_mode(s)
Us2,Ss2,Vs2 = svd(s2)

w = [v[(i-1)*np+1:i*np] for i = 1:nt]
b = ParamArray.(w)
t = Snapshots(b,r)
Ut,St,Vt = svd(t)
t2 = RB.change_mode(t)
Ut2,St2,Vt2 = svd(t2)

A = stack(v)
UA,SA,VA = svd(A)
x = map(1:np) do ip
  hcat(v[ip:np:nt*np]...)'
end
B = hcat(x...)
UB,SB,VB = svd(B)

@check Ut ≈ Us ≈ UA
@check St ≈ Ss ≈ SA
@check Ut2 ≈ Us2 ≈ UB
@check St2 ≈ Ss2 ≈ SB

v1 = A[:,rand(axes(A,2))]
w1 = B[:,rand(axes(B,2))]

@check norm(UA*UA'*v1 - v1)/sqrt(ns) ≤ 1e-12
@check norm(UB*UB'*w1 - w1)/sqrt(nt) ≤ 1e-12

nparts = 2
nrowsA = floor(Int,ns/nparts)
A_parts = [A[(i-1)*nrows+1:i*nrows,:] for i = 1:nparts]
v1_parts = [v1[(i-1)*nrows+1:i*nrows] for i = 1:nparts]

UA_parts = map(A_parts) do A
  U,V,S = svd(A)
  U
end

solA = hcat(UA[:,1],vcat(map(x->x[:,1],UA_parts)...))
plot(solA)

v1_rec_parts = map(UA_parts,v1_parts) do U,v1
  U*U'*v1
end

@check norm(vcat(v1_rec_parts...) - v1)/sqrt(ns) ≤ 1e-12
