using Gridap
using Gridap.FESpaces
using ForwardDiff
using LinearAlgebra
using Test
using Gridap.FESpaces: get_algebraic_operator
using Mabla.FEM
using Mabla.RB
using Mabla.Distributed
using Gridap.ODEs
using Gridap.ODEs.TransientFETools
using Gridap.ODEs.ODETools
using Gridap.Helpers
using Gridap.MultiField
using GridapDistributed
using PartitionedArrays
using DrWatson

θ = 1
dt = 0.01
t0 = 0.0
tf = 0.1

pranges = fill([0,1],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)
r = realization(ptspace,nparams=3)
μ = FEM._get_params(r)[1]

domain = (0,1,0,1)
mesh_partition = (2,2)
mesh_cells = (4,4)
ranks = with_debug() do distribute
  distribute(LinearIndices((prod(4),)))
end
model = CartesianDiscreteModel(ranks,mesh_partition,domain,mesh_cells)

a(x,μ,t) = μ[1] + μ[2]*sin(2*π*t/μ[3])
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

f(x,μ,t) = sin(π*t/μ[3])
f(μ,t) = x->f(x,μ,t)
fμt(μ,t) = TransientParamFunction(f,μ,t)

g(x,μ,t) = 0.0
g(μ,t) = x->g(x,μ,t)
gμt(μ,t) = TransientParamFunction(g,μ,t)

u0(x,μ) = 0
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)

b(μ,t,v) = ∫(fμt(μ,t)*v)dΩ
a(μ,t,du,v) = ∫(aμt(μ,t)*∇(v)⋅∇(du))dΩ
m(μ,t,dut,v) = ∫(v*dut)dΩ

order = 1
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

T = Float64
reffe = ReferenceFE(lagrangian,T,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags="boundary")
trial = TransientTrialParamFESpace(test,gμt)
feop = AffineTransientParamFEOperator(m,a,b,ptspace,trial,test)

uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

# sol = solve(fesolver,feop,uh0μ,r)

# for (uh,rt) in sol
# end

# # gridap

# _a(x,t) = a(x,μ,t)
# _a(t) = x->_a(x,t)

# _f(x,t) = f(x,μ,t)
# _f(t) = x->_f(x,t)

# _g(x,t) = g(x,μ,t)
# _g(t) = x->_g(x,t)

# _b(t,v) = ∫(_f(t)*v)dΩ
# _a(t,du,v) = ∫(_a(t)*∇(v)⋅∇(du))dΩ
# _m(t,dut,v) = ∫(v*dut)dΩ

# _trial = TransientTrialFESpace(test,_g)
# _feop = TransientAffineFEOperator(_m,_a,_b,_trial,test)
# _u0 = interpolate_everywhere(x->0.0,_trial(0.0))

# _sol = solve(fesolver,_feop,_u0,t0,tf)

# for ((uh,rt),(_uh,_t)) in zip(sol,_sol)
#   t = get_times(rt)
#   @check t ≈ _t "$t != $_t"
#   map(local_views(uh),local_views(_uh)) do uh,_uh
#     uh1 = FEM._getindex(uh,1)
#     @check get_free_dof_values(uh1) ≈ get_free_dof_values(_uh) "$(get_free_dof_values(uh1)) != $(get_free_dof_values(_uh))"
#     @check uh1.dirichlet_values ≈ _uh.dirichlet_values
#   end
# end

dir = datadir("distr_toy_heateq")
info = RBInfo(dir;nsnaps_state=5,nsnaps_test=0)

# nparams = RB.num_params(info)
# sol = solve(fesolver,feop,uh0μ;nparams)
# odesol = sol.odesol
# r = odesol.r
# stats = @timed begin
#   vals,initial_vals = collect(odesol)
# end
# # Snapshots(vals,initial_vals,r)
# V = ParamVector{Float64, Vector{Vector{Float64}}, 10}
# snaps = map(local_views(initial_vals),initial_vals.index_partition) do ival_part,iip
#   part = part_id(iip)
#   for (k,v) in enumerate(vals)
#     vals_part = Vector{V}(undef,length(vals))
#     map(local_views(v),v.index_partition) do val,ip
#       if part_id(ip) == part
#         vals_part[k] = val
#       end
#     end
#   end
#   Snapshots(vals_part,ival_part,r)
# end

snaps,comp = RB.collect_solutions(info,fesolver,feop,uh0μ)
bs,bt = reduced_basis(info,feop,snaps)

s1,s̃1 = map(local_views(snaps),local_views(bs)) do snaps,bs
  values = snaps.values
  vecs = map(eachindex(values)) do i
    hcat(values[i].array...)
  end
  M = hcat(vecs...)
  @assert M ≈ snaps

  U1 = tpod(snaps)
  U2 = tpod(M)
  @assert U1 ≈ U2
  s̃ = RB.recast_compress(U1,snaps)
  @assert norm(s̃ - snaps) / norm(snaps) < 10*RB.get_tol(info)

  s1 = ParamArray([snaps[:,k] for k = 1:num_params(snaps):size(snaps,2)])
  s̃1 = ParamArray([s̃[:,k] for k = 1:num_params(s̃):size(s̃,2)])
  s1,s̃1
end |> tuple_of_arrays

# RB._plot(trial,s̃,dir=dir)

# very ugly
x = get_free_dof_values(zero(test))
index_partition = x.index_partition
pv1 = PVector(local_views(s1),index_partition)
p̃v1 = PVector(local_views(s̃1),index_partition)
norm(p̃v1 - pv1) / norm(pv1)

get_realization(s::AbstractTransientSnapshots) = s.realization
allr = map(get_realization,local_views(snaps))
r = PartitionedArrays.getany(allr)[1,:]
times = get_times(r)
for (it,t) = enumerate(times)
  rt = FEM.get_at_time(r,t)
  free_values = PVector(map(x->x[it],local_views(p̃v1)),index_partition)
  sht = FEFunction(test,free_values)
  writevtk(Ω,dir*"/solution_$t"*".vtu",cellfields=["u"=>sht])
end


μ = FEM._get_params(realization(feop))[1]
_a(x,t) = a(x,μ,t)
_a(t) = x->_a(x,t)

_f(x,t) = f(x,μ,t)
_f(t) = x->_f(x,t)

_g(x,t) = g(x,μ,t)
_g(t) = x->_g(x,t)

_b(t,v) = ∫(_f(t)*v)dΩ
_a(t,du,v) = ∫(_a(t)*∇(v)⋅∇(du))dΩ
_m(t,dut,v) = ∫(v*dut)dΩ

_trial = TransientTrialFESpace(test,_g)
_feop = TransientAffineFEOperator(_m,_a,_b,_trial,test)
_u0 = interpolate_everywhere(x->0.0,_trial(0.0))

_sol = solve(fesolver,_feop,_u0,t0,tf)

dir = datadir("boh")
i_am_main(ranks) && !isdir(dir) && mkdir(dir)
for (uₕ,t) in _sol
  println(" > Computing solution at time $t")
  file = dir*"/solution_$t"*".vtu"
  writevtk(Ω,file,cellfields=["u"=>uₕ])
end

(uₕ,t),_ = iterate(_sol)
file = dir*"/solution_$t"*".vtu"
writevtk(Ω,file,cellfields=["u"=>uₕ])

parts=get_parts(Ω)
map(visualization_data(arg,args...;kwargs...)) do visdata
  write_vtk_file(
  parts,visdata.grid,visdata.filebase,celldata=visdata.celldata,nodaldata=visdata.nodaldata)
end
