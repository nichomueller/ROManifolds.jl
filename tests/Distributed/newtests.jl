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

pranges = fill([1,10],3)
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

order = 1
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

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

res(μ,t,u,v,dΩ) = ∫(v*∂t(u))dΩ + ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ - ∫(fμt(μ,t)*v)dΩ
jac(μ,t,u,du,v,dΩ) = ∫(aμt(μ,t)*∇(v)⋅∇(du))dΩ
jac_t(μ,t,u,dut,v,dΩ) = ∫(v*dut)dΩ

trian_res = (Ω,)
trian_jac = (Ω,)
trian_jac_t = (Ω,)

T = Float64
reffe = ReferenceFE(lagrangian,T,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags="boundary")
trial = TransientTrialParamFESpace(test,gμt)
feop = AffineTransientParamFEOperator(res,jac,jac_t,ptspace,trial,test,trian_res,trian_jac,trian_jac_t)

uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

dir = datadir("distr_toy_heateq")
info = RBInfo(dir;nsnaps_state=10,nsnaps_test=5,save_structures=false)

rbsolver = RBSolver(info,fesolver)

snaps,comp = fe_solutions(rbsolver,feop,uh0μ)

function load_distributed_snapshots(info::RBInfo)
  i_filename = Distributed.get_ind_part_filename(info)
  s_filename = RB.get_snapshots_filename(info)
  map(readdir(info.dir;join=true)) do dir
    part = parse(Int,dir[end])
    i_part_filename = Distributed.get_part_filename(i_filename,part)
    s_part_filename = Distributed.get_part_filename(s_filename,part)
    deserialize(i_part_filename),deserialize(s_part_filename)
  end |> tuple_of_arrays
end

aa,bb=load_distributed_snapshots(info)

ip = with_debug() do distribute
  i_filename = Distributed.get_ind_part_filename(info)
  ip = map(readdir(info.dir;join=true)) do dir
    part = parse(Int,dir[end])
    i_part_filename = Distributed.get_part_filename(i_filename,part)
    deserialize(i_part_filename)
  end
  distribute(ip)
end

_s = with_debug() do distribute
  Distributed.load_distributed_snapshots(distribute,info)
end
STOP
# nparams = num_params(info)
# sol = solve(fesolver,feop,uh0μ;nparams)
# odesol = sol.odesol
# r = odesol.r

# stats = @timed begin
#   vals = collect(odesol)
# end
# # snaps = Snapshots(vals,r)
# getV(::AbstractVector{<:PVector{V}}) where V = V
# index_partition = first(vals).index_partition
# parts = map(part_id,index_partition)
# snaps = map(parts) do part
#   vals_part = Vector{getV(vals)}(undef,length(vals))
#   for (k,v) in enumerate(vals)
#     map(local_views(v),index_partition) do val,ip
#       if part_id(ip) == part
#         vals_part[k] = val
#       end
#     end
#   end
#   Snapshots(vals_part,r)
# end
# ciao = PVector(snaps,index_partition)
# DistributedSnapshots(ciao)

# map(local_views(ciao)) do boh
#   typeof(boh)
# end

# red_op = reduced_operator(rbsolver,feop,snaps)
red_trial,red_test = reduced_fe_space(info,feop,snaps)
