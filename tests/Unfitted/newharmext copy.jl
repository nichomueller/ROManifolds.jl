using Gridap
using Gridap.FESpaces
using Gridap.Algebra
using Gridap.CellData
using Gridap.Fields
using Gridap.Arrays
using Gridap.MultiField
using GridapEmbedded
using GridapEmbedded.Interfaces
using ReducedOrderModels
using ReducedOrderModels.Utils
using DrWatson
using BlockArrays

# ################################################################################

# struct MyMask <: Map end

# function Arrays.return_cache(::MyMask,cellarray,cellids)
#   z = zero(eltype(cellarray))
#   a = similar(cellarray)
#   return z,a
# end

# function Arrays.evaluate!(cache,::MyMask,cellarray,cellids)
#   z,a = cache
#   for (is,i) in enumerate(cellids)
#     if i != 0
#       a[is] = z
#     else
#       a[is] = cellarray[is]
#     end
#   end
#   a
# end

# struct MyIdsMask <: Map end

# function Arrays.return_cache(::MyIdsMask,cellids_with_zeros,cellids)
#   z = zero(eltype(cellids_with_zeros))
#   a = similar(cellids_with_zeros)
#   return z,a
# end

# function Arrays.evaluate!(cache,::MyIdsMask,cellids_with_zeros,cellids)
#   z,a = cache
#   for (is,i) in enumerate(cellids_with_zeros)
#     if i != 0
#       a[is] = z
#     else
#       a[is] = cellids[is]
#     end
#   end
#   a
# end

# ################################################################################

u(x) = x[1] - x[2]
f(x) = -Δ(u)(x)
ud(x) = u(x)

R = 0.5
L = 0.8*(2*R)
p1 = Point(0.0,0.0)
p2 = p1 + VectorValue(L,0.0)

geo1 = disk(R,x0=p1)
geo2 = disk(R,x0=p2)
geo = setdiff(geo1,geo2)
geo_out = !geo

t = 1.01
pmin = p1-t*R
pmax = p1+t*R

n = 30
partition = (n,n)
model = CartesianDiscreteModel(pmin,pmax,partition)
dp = pmax - pmin
h = dp[1]/n

cutgeo = cut(model,geo)
cutgeo_out = cut(model,geo_out)

Ω_bg = Triangulation(model)
Ω_act = Triangulation(cutgeo,ACTIVE)
Ω_act_out = Triangulation(cutgeo,ACTIVE_OUT)
Ω = Triangulation(cutgeo,PHYSICAL)
Ω_out = Triangulation(cutgeo,PHYSICAL_OUT)
Γ = EmbeddedBoundary(cutgeo)
n_Γ = get_normal_vector(Γ)
Γ_out = EmbeddedBoundary(cutgeo_out)
n_Γ_out = get_normal_vector(Γ_out)

order = 1
degree = 2*order
dΩ_bg = Measure(Ω_bg,degree)
dΩ = Measure(Ω,degree)
dΩ_out = Measure(Ω_out,degree)
dΓ = Measure(Γ,degree)
dΓ_out = Measure(Γ_out,degree)

reffe = ReferenceFE(lagrangian,Float64,order)

Vcut = FESpace(Ω_act,reffe;conformity=:H1)
Ucut = TrialFESpace(Vcut)

V = FESpace(Ω_bg,reffe;conformity=:H1)
U = TrialFESpace(V)

γd = 10.0

acut(u,v) = ∫( ∇(v)⋅∇(u) )*dΩ + ∫( (γd/h)*v*u - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u ) * dΓ
aout(u,v) = ∫( ∇(v)⋅∇(u) )*dΩ_out
a(u,v) = acut(u,v) + aout(u,v)

lcut(v) = ∫( v*f ) * dΩ + ∫( (γd/h)*v*ud - (n_Γ⋅∇(v))*ud ) * dΓ
lout(v) = ∫( ∇(v)⋅∇(ud) )*dΩ_out
l(v) = lcut(v) + lout(v)

op = AffineFEOperator(a,l,U,V)
uh = solve(op)

opcut = AffineFEOperator(acut,lcut,Ucut,Vcut)
uhcut = solve(opcut)

writevtk(Ω_bg,datadir("plts/sol"),cellfields=["uh"=>uh])
writevtk(Ω,datadir("plts/sol_ok"),cellfields=["uh"=>uhcut])
writevtk(Ω,datadir("plts/sol_err"),cellfields=["e"=>uh-uhcut])

# a_cut_cut(u,v) = ∫( ∇(v)⋅∇(u) ) * dΩ + ∫( (γd/h)*v*u - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u ) * dΓ
# a_cut_out(u,v) = ∫( (γd/h)*v*u - v*(n_Γ_out⋅∇(u)) - (n_Γ_out⋅∇(v))*u ) * dΓ_out
# a_out_out(u,v) = ∫( ∇(v)⋅∇(u) ) * dΩ_out + ∫( (γd/h)*v*u - v*(n_Γ_out⋅∇(u)) - (n_Γ_out⋅∇(v))*u ) * dΓ_out
# a((ucut,uout),(vcut,vout)) = a_cut_cut(ucut,vcut) + a_cut_out(ucut,vout) + a_out_out(uout,vout)

# l_cut(v) = ∫( v*f ) * dΩ + ∫( (γd/h)*v*ud - (n_Γ⋅∇(v))*ud ) * dΓ
# l_out(v) = ∫( v*0 ) * dΩ_out #+ ∫( (γd/h)*v*ud - (n_Γ_out⋅∇(v))*ud ) * dΓ_out
# l((vcut,vout)) = l_cut(vcut) + l_out(vout)

# op = AffineFEOperator(a,l,U,V)
# uh = solve(op)

# opcut = AffineFEOperator(a_cut_cut,l_cut,Ucut,Vcut)
# uhcut = solve(opcut)

# norm(uh.free_values[Block(1)]) ≈ norm(uhcut.free_values)
