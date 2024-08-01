using Pkg; Pkg.activate(".")

using Gridap
using Test
using DrWatson
using Serialization

using Mabla.FEM
using Mabla.FEM.TProduct
using Mabla.FEM.ParamDataStructures
using Mabla.FEM.ParamFESpaces
using Mabla.FEM.ParamSteady
using Mabla.FEM.ParamODEs

using Mabla.RB
using Mabla.RB.RBSteady
using Mabla.RB.RBTransient

# time marching
θ = 0.5
dt = 0.01
t0 = 0.0
tf = 0.1

# parametric space
pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

# geometry
n = 100
domain = (0,1,0,1)
partition = (n,n)
model = TProductModel(domain,partition)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet",[1,2,3,4,5,6,8])
add_tag_from_tags!(labels,"neumann",[7])

order = 2
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
Γn = BoundaryTriangulation(model,tags=["neumann"])
dΓn = Measure(Γn,degree)

# weak formulation
a(x,μ,t) = 1+exp(-sin(t)^2*x[1]/sum(μ))
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

f(x,μ,t) = 1.
f(μ,t) = x->f(x,μ,t)
fμt(μ,t) = TransientParamFunction(f,μ,t)

h(x,μ,t) = abs(cos(t/μ[3]))
h(μ,t) = x->h(x,μ,t)
hμt(μ,t) = TransientParamFunction(h,μ,t)

g(x,μ,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
g(μ,t) = x->g(x,μ,t)
gμt(μ,t) = TransientParamFunction(g,μ,t)

u0(x,μ) = 0
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)

stiffness(μ,t,u,v,dΩ) = ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ
mass(μ,t,uₜ,v,dΩ) = ∫(v*uₜ)dΩ
rhs(μ,t,v,dΩ,dΓn) = ∫(fμt(μ,t)*v)dΩ + ∫(hμt(μ,t)*v)dΓn
res(μ,t,u,v,dΩ,dΓn) = mass(μ,t,∂t(u),v,dΩ) + stiffness(μ,t,u,v,dΩ) - rhs(μ,t,v,dΩ,dΓn)

trian_res = (Ω.trian,Γn)
trian_stiffness = (Ω.trian,)
trian_mass = (Ω.trian,)

induced_norm(du,v) = (∫(du*v)dΩ + ∫(∇(v)⋅∇(du))dΩ)/dt

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = TransientTrialParamFESpace(test,gμt)
feop = TransientParamLinearFEOperator((stiffness,mass),res,induced_norm,ptspace,
  trial,test,trian_res,trian_stiffness,trian_mass)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

fesolver = ThetaMethod(LUSolver(),dt,θ)
ϵ = 1e-4
rbsolver = RBSolver(fesolver,ϵ;nsnaps_state=50,nsnaps_test=10,nsnaps_mdeim=20)
test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("heateq","elasticity_h1")))

fesnaps,festats = fe_solutions(rbsolver,feop,uh0μ)
rbop = reduced_operator(rbsolver,feop,fesnaps)
rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps)
results = rb_results(rbsolver,rbop,fesnaps,rbsnaps,festats,rbstats)

println(compute_error(results))
println(compute_speedup(results))

save(test_dir,fesnaps)
save(test_dir,rbop)

X = assemble_norm_matrix(feop)
X1 = X.arrays_1d[1] + X.gradients_1d[1]
X2 = X.arrays_1d[2] + X.gradients_1d[2]
Xk = kron(X)

s = select_snapshots(fesnaps,1:20)

using LinearAlgebra
using SparseArrays

mat = s
sizes = size(s)
Φ = Array{Float64,3}[]

mat_k = reshape(mat,sizes[1],:)
Ur,Σr,Vr = RBSteady._tpod(mat_k,X1)
push!(Φ,reshape(Ur,1,sizes[1],size(Ur,2)))
mat_k = reshape(Σr.*Vr',size(Ur,2)*sizes[2],:)
# X1_hat = Ur'*X1*Ur
X2′ = kron(X2,Float64.(I(size(Ur,2))))
C = cholesky(X2′)
L,p = sparse(C.L),C.p
Xmat = L'*mat_k[p,:]
Ũr,Σr,Vr = RBSteady.truncated_svd(Xmat)
Ur = (L'\Ũr)[invperm(p),:]
push!(Φ,reshape(Ur,size(Φ[1],3),sizes[2],size(Ur,2)))

mat_k = reshape(Σr.*Vr',size(Ur,2)*sizes[3],:)
Ur,Σr,Vr = RBSteady._tpod(mat_k)
push!(Φ,reshape(Ur,size(Φ[2],3),sizes[3],size(Ur,2)))
mat = reshape(Σr.*Vr',size(Ur,2),sizes[3],:)

bs = cores2basis(Φ[1],Φ[2])
X′ = kron(X2,X1)
bs'*X′*bs

# re orthogonalize wrt Xk
H′ = cholesky(X′)
L′,p′ = sparse(H′.L),H′.p
H = cholesky(Xk)
L,p = sparse(H.L),H.p

# # bs = H\H′*bs′
# bs_temp = L′'*bs′[p′,:]
# # Bs = L'\bs_temp[invperm(p),:]
# Q,R = RBSteady.pivoted_qr(L'*bs_temp[p,:])
# Bs = (L'\Q)[invperm(p),:]
# Bs'*Xk*Bs

sflat = reshape(s,size(s,1)*size(s,2),size(s,3)*size(s,4))
s′ = reshape(L′'\(L'*sflat[p,:])[invperm(p′),:],size(s))
mat = s′
Φ′ = Array{Float64,3}[]

mat_k = reshape(mat,sizes[1],:)
Ur,Σr,Vr = RBSteady._tpod(mat_k,X1)
push!(Φ′,reshape(Ur,1,sizes[1],size(Ur,2)))
mat_k = reshape(Σr.*Vr',size(Ur,2)*sizes[2],:)
# X1_hat = Ur'*X1*Ur
X2′ = kron(X2,Float64.(I(size(Ur,2))))
c = cholesky(X2′)
l,q = sparse(c.L),c.p
Xmat = l'*mat_k[q,:]
Ũr,Σr,Vr = RBSteady.truncated_svd(Xmat)
Ur = (l'\Ũr)[invperm(q),:]
push!(Φ′,reshape(Ur,size(Φ′[1],3),sizes[2],size(Ur,2)))

mat_k = reshape(Σr.*Vr',size(Ur,2)*sizes[3],:)
Ur,Σr,Vr = RBSteady._tpod(mat_k)
push!(Φ′,reshape(Ur,size(Φ′[2],3),sizes[3],size(Ur,2)))
mat = reshape(Σr.*Vr',size(Ur,2),sizes[3],:)

bs′ = cores2basis(Φ′[1],Φ′[2])
bs′'*X′*bs′

bs′′ = L'\(L′'*cores2basis(Φ′[1],Φ′[2])[p′,:])[invperm(p),:]
bs′′'*Xk*bs′′

Y = L′'*cores2basis(Φ′[1],Φ′[2])[p′,:]
Y'*Y # orthogonal
Z = L'\Y[invperm(p),:]#[invperm(p),:] # should be Xk orthogonal
Z'*Xk[p,p]*Z
Z[invperm(p),:]'*Xk*Z[invperm(p),:]

u1 = reshape(select_snapshots(fesnaps,1,1),:,1)

v = u1 - Z[invperm(p),:]*Z[invperm(p),:]'Xk*u1
vnorm = v'*Xk*v
bok = cores2basis(b.cores_space...)
vok = u1 - bok*bok'Xk*u1
vnormok = vok'*Xk*vok

ϕs = truncated_pod(sflat,Xk)
vok = u1 - ϕs*ϕs'Xk*u1
vnormok = vok'*Xk*vok

# what if we consider X′ = 1/h^2 M?
δ = 1/20
M1 = X.arrays_1d[1] / δ
M2 = X.arrays_1d[2] / δ
X′ = kron(M2,M1)
H′ = cholesky(X′)
L′,p′ = sparse(H′.L),H′.p

sflat = reshape(s,size(s,1)*size(s,2),size(s,3)*size(s,4))
s′ = reshape(L′'\(L'*sflat[p,:])[invperm(p′),:],size(s))
mat = s′
Φ′ = Array{Float64,3}[]

mat_k = reshape(mat,sizes[1],:)
Ur,Σr,Vr = RBSteady._tpod(mat_k,M1)
push!(Φ′,reshape(Ur,1,sizes[1],size(Ur,2)))
mat_k = reshape(Σr.*Vr',size(Ur,2)*sizes[2],:)
X2′ = kron(M2,Float64.(I(size(Ur,2))))
c = cholesky(X2′)
l,q = sparse(c.L),c.p
Xmat = l'*mat_k[q,:]
Ũr,Σr,Vr = RBSteady.truncated_svd(Xmat)
Ur = (l'\Ũr)[invperm(q),:]
push!(Φ′,reshape(Ur,size(Φ′[1],3),sizes[2],size(Ur,2)))

mat_k = reshape(Σr.*Vr',size(Ur,2)*sizes[3],:)
Ur,Σr,Vr = RBSteady._tpod(mat_k)
push!(Φ′,reshape(Ur,size(Φ′[2],3),sizes[3],size(Ur,2)))
mat = reshape(Σr.*Vr',size(Ur,2),sizes[3],:)

bs′ = cores2basis(Φ′[1],Φ′[2])
bs′'*X′*bs′
v = u1 - bs′*bs′'X′*u1
vnorm = v'*X′*v

Y = L′'*cores2basis(Φ′[1],Φ′[2])[p′,:]
Y'*Y # orthogonal
Z = L'\Y[invperm(p),:]#[invperm(p),:] # should be Xk orthogonal
Z'*Xk[p,p]*Z
Z[invperm(p),:]'*Xk*Z[invperm(p),:]

v = u1 - Z[invperm(p),:]*Z[invperm(p),:]'Xk*u1
vnorm = v'*Xk*v

bl2 = ttsvd(s)
bsl2 = cores2basis(bl2[1],bl2[2])
vl2 = u1 - bsl2*bsl2'*u1

X′ = kron(M2,M1)
H′ = cholesky(X′)
L′,p′ = sparse(H′.L),H′.p
Xk = kron(X)
H = cholesky(Xk)
L,p = sparse(H.L),H.p

MX = TProductArray([X.arrays_1d[1]/δ,X.arrays_1d[2]/δ])
MK = kron(MX)
sl2 = reshape(L′'\(L'*sflat[p,:])[invperm(p′),:],size(s))
bl2 = ttsvd(s,MX)
bsl2 = cores2basis(bl2[1],bl2[2])
vl2 = u1 - bsl2*bsl2'*MK*u1
vl2 = u1 - bsl2*bsl2'*Xk*u1

Y = L′'*bsl2[p′,:]
Z = L'\Y[invperm(p),:]
v = u1 - Z[invperm(p),:]*Z[invperm(p),:]'Xk*u1
v = u1[p] - Z[p,:]*Z[p,:]'Xk[p,p]*u1[p]

bh1 = ttsvd(s,X)
bsh1 = cores2basis(bh1[1],bh1[2])
vh1 = u1 - bsh1*bsh1'*Xk*u1
vh1'*Xk*vh1
