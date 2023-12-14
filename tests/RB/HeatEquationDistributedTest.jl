# using Gridap
# using GridapDistributed
# using PartitionedArrays

# function main(ranks)
#   domain = (0,1,0,1)
#   mesh_partition = (2,2)
#   mesh_cells = (4,4)
#   model = CartesianDiscreteModel(ranks,mesh_partition,domain,mesh_cells)
#   order = 2
#   u((x,y)) = (x+y)^order
#   f(x) = -Δ(u,x)
#   reffe = ReferenceFE(lagrangian,Float64,order)
#   V = TestFESpace(model,reffe,dirichlet_tags="boundary")
#   U = TrialFESpace(u,V)
#   Ω = Triangulation(model)
#   dΩ = Measure(Ω,2*order)
#   a(u,v) = ∫( ∇(v)⋅∇(u) )dΩ
#   l(v) = ∫( v*f )dΩ
#   op = AffineFEOperator(a,l,U,V)
#   uh = solve(op)
#   writevtk(Ω,"results",cellfields=["uh"=>uh,"grad_uh"=>∇(uh)])
# end

# with_debug() do distribute
#   ranks = distribute(LinearIndices((4,)))
#   main(ranks)
# end

using Gridap
using Gridap.FESpaces
using GridapDistributed
using PartitionedArrays
using Mabla.FEM
using Mabla.RB
using Mabla.Distributed
import UnPack: @unpack

root = pwd()
test_path = "$root/results/HeatEquation/cube_2x2.json"
ϵ = 1e-4
load_solutions = true
save_solutions = true
load_structures = false
save_structures = true
postprocess = true
norm_style = :H1
nsnaps_state = 50
nsnaps_mdeim = 20
nsnaps_test = 10
st_mdeim = true
rbinfo = RBInfo(test_path;ϵ,norm_style,nsnaps_state,nsnaps_mdeim,nsnaps_test,st_mdeim)

domain = (0,1,0,1)
mesh_partition = (2,2)
mesh_cells = (4,4)

# with_debug() do distribute
#   ranks = distribute(LinearIndices((4,)))
#   model = CartesianDiscreteModel(ranks,mesh_partition,domain,mesh_cells)

#   order = 1
#   degree = 2*order
#   Ω = Triangulation(model)
#   Γn = BoundaryTriangulation(model,tags=[7,8])
#   dΩ = Measure(Ω,degree)
#   dΓn = Measure(Γn,degree)

#   ranges = fill([1.,10.],3)
#   pspace = PSpace(ranges)

#   a(x,μ,t) = exp((sin(t)+cos(t))*x[1]/sum(μ))
#   a(μ,t) = x->a(x,μ,t)
#   aμt(μ,t) = PTFunction(a,μ,t)

#   f(x,μ,t) = 1.
#   f(μ,t) = x->f(x,μ,t)
#   fμt(μ,t) = PTFunction(f,μ,t)

#   h(x,μ,t) = abs(cos(t/μ[3]))
#   h(μ,t) = x->h(x,μ,t)
#   hμt(μ,t) = PTFunction(h,μ,t)

#   g(x,μ,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
#   g(μ,t) = x->g(x,μ,t)

#   u0(x,μ) = 0
#   u0(μ) = x->u0(x,μ)
#   u0μ(μ) = PFunction(u0,μ)

#   res(μ,t,u,v) = ∫ₚ(v*∂ₚt(u),dΩ) + ∫ₚ(aμt(μ,t)*∇(v)⋅∇(u),dΩ) - ∫ₚ(fμt(μ,t)*v,dΩ) - ∫ₚ(hμt(μ,t)*v,dΓn)
#   jac(μ,t,u,du,v) = ∫ₚ(aμt(μ,t)*∇(v)⋅∇(du),dΩ)
#   jac_t(μ,t,u,dut,v) = ∫ₚ(v*dut,dΩ)

#   T = Float64
#   reffe = ReferenceFE(lagrangian,T,order)
#   test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6])
#   trial = TransientTrialPFESpace(test,g)
#   feop = AffinePTFEOperator(res,jac,jac_t,pspace,trial,test)
#   t0,tf,dt,θ = 0.,0.3,0.005,0.5
#   uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
#   fe = trial(Table([rand(3) for _ = 1:3]),[t0,dt])
#   println(typeof(fe))
#   fesolver = PThetaMethod(LUSolver(),uh0μ,θ,dt,t0,tf)
#   uh0μ(Table([rand(3) for _ = 1:3]))
#   error("stop")

#   sols,params,stats = collect_solutions(rbinfo,fesolver,feop)
#   # rbspace = reduced_basis(rbinfo,feop,sols)
#   # rbrhs,rblhs = collect_compress_rhs_lhs(rbinfo,feop,fesolver,rbspace,params)
# end

# function Base.getindex(
#   a::JaggedArray{Vector{<:AbstractArray}},
#   i::Integer)
#   JaggedArray(a.data[i],a.ptrs)
# end

# function Base.getindex(
#   cache::PartitionedArrays.VectorAssemblyCache{Vector{<:AbstractArray}},
#   i::Integer)

#   @unpack (neighbors_snd,neighbors_rcv,local_indices_snd,local_indices_rcv,
#           buffer_snd,buffer_rcv) = cache

#   PartitionedArrays.VectorAssemblyCache(
#     neighbors_snd,
#     neighbors_rcv,
#     local_indices_snd,
#     local_indices_rcv,
#     buffer_snd[i],
#     buffer_rcv[i])
# end

# Base.length(a::MPIArray{<:PTArray}) = first(size(a))
# Base.length(a::DebugArray{<:PTArray}) = first(size(a))
# Base.eachindex(a::MPIArray{<:PTArray}) = Base.OneTo(length(a))
# Base.eachindex(a::DebugArray{<:PTArray}) = Base.OneTo(length(a))

# function PartitionedArrays.assemble_impl!(
#   f,
#   vector_partition::AbstractArray{<:PTArray},
#   cache,
#   ::Type{PartitionedArrays.VectorAssemblyCache{S}}) where S

#   println("CORRECT assemble_impl!")
#   elS = eltype(S)
#   elC = PartitionedArrays.VectorAssemblyCache{elS}
#   map(vector_partition,cache) do vector_partition,cache
#     assemble_impl!(f,vector_partition[i],cache[i],elC)
#   end
#   # for i in eachindex(vector_partition)
#   #   assemble_impl!(f,vector_partition[i],cache[i],elC)
#   # end
# end

# function PartitionedArrays.assemble_impl!(
#   f,
#   vector_partition::AbstractArray{<:PTArray},
#   cache,
#   ::Type{<:PartitionedArrays.VectorAssemblyCache})

#   buffer_snd = map(vector_partition,cache) do values,cache
#     local_indices_snd = cache.local_indices_snd
#     # for i in eachindex(values)
#     #   for (p,lid) in enumerate(local_indices_snd.data)
#     #     cache.buffer_snd.data[i][p] = values[i][lid]
#     #   end
#     # end
#     println(typeof(cache.buffer_snd.data))
#     println(typeof(values))
#     for (p,lid) in enumerate(local_indices_snd.data)
#       cache.buffer_snd.data[p] = values[lid]
#     end
#     cache.buffer_snd
#   end
#   neighbors_snd,neighbors_rcv,buffer_rcv = map(cache) do cache
#     cache.neighbors_snd,cache.neighbors_rcv,cache.buffer_rcv
#   end |> tuple_of_arrays

#   graph = ExchangeGraph(neighbors_snd,neighbors_rcv)
#   println(typeof(buffer_rcv))
#   println(typeof(buffer_snd))
#   t = exchange!(buffer_rcv,buffer_snd,graph)
#   # Fill values from rcv buffer asynchronously
#   @async begin
#     wait(t)
#     map(vector_partition,cache) do values,cache
#       local_indices_rcv = cache.local_indices_rcv
#       for i in eachindex(values)
#         for (p,lid) in enumerate(local_indices_rcv.data)
#           values[lid] = f(values[i][lid],cache.buffer_rcv.data[i][p])
#         end
#       end
#     end
#     nothing
#   end
# end

# function _exchange_impl!(rcv,snd,graph,::Type{T}) where T<:AbstractVector
#   @assert is_consistent(graph)
#   @assert eltype(rcv) <: JaggedArray
#   snd_ids = graph.snd
#   rcv_ids = graph.rcv
#   @assert length(rcv_ids) == length(rcv)
#   @assert length(rcv_ids) == length(snd)
#   #for k = eachindex(rcv)
#     for rcv_id in eachindex(rcv_ids)
#       for (i,snd_id) in enumerate(rcv_ids[rcv_id])
#         snd_snd_id = JaggedArray(snd[snd_id])
#         j = first(findall(k->k==rcv_id,snd_ids[snd_id]))
#         ptrs_rcv = rcv[rcv_id].ptrs
#         ptrs_snd = snd_snd_id.ptrs
#         @assert ptrs_rcv[i+1]-ptrs_rcv[i] == ptrs_snd[j+1]-ptrs_snd[j]
#         for p in 1:(ptrs_rcv[i+1]-ptrs_rcv[i])
#           p_rcv = p+ptrs_rcv[i]-1
#           p_snd = p+ptrs_snd[j]-1
#           rcv[rcv_id].data[p_rcv] = snd_snd_id.data[p_snd]
#         end
#       end
#     end
#   #end
#   @async rcv
# end

ranks = LinearIndices((4,))
model = CartesianDiscreteModel(ranks,mesh_partition,domain,mesh_cells)
order = 1

g(x,μ,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
g(μ,t) = x->g(x,μ,t)
u0(x,μ) = 0
u0(μ) = x->u0(x,μ)
u0μ(μ) = PFunction(u0,μ)

T = Float64
reffe = ReferenceFE(lagrangian,T,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6])
trial = TransientTrialPFESpace(test,g)
t0 = 0.
μ = Table([rand(3) for _ = 1:3])
u,f = u0μ(μ),trial(μ,t0)
free_values = zero_free_values(f)
dirichlet_values = get_dirichlet_dof_values(f)
map(f.spaces,local_views(free_values),dirichlet_values) do V,fvec,dvec
  interpolate_everywhere!(u,fvec,dvec,V)
end
a = free_values
# consistent!(a) |> wait
insert(a,b) = b
cache = map(reverse,a.cache)
t = assemble!(insert,partition(a),cache)

# with_debug() do distribute
#   ranks = distribute(LinearIndices((4,)))
#   model = CartesianDiscreteModel(ranks,mesh_partition,domain,mesh_cells)
#   order = 1

#   g(x,μ,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
#   g(μ,t) = x->g(x,μ,t)
#   u0(x,μ) = 0
#   u0(μ) = x->u0(x,μ)
#   u0μ(μ) = PFunction(u0,μ)

#   T = Float64
#   reffe = ReferenceFE(lagrangian,T,order)
#   test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6])
#   trial = TransientTrialPFESpace(test,g)
#   t0 = 0.
#   uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
#   uh0μ(Table([rand(3) for _ = 1:3]))
# end

# with_debug() do distribute
#   ranks = distribute(LinearIndices((4,)))
#   model = CartesianDiscreteModel(ranks,mesh_partition,domain,mesh_cells)
#   order = 1

#   g(x,t) = exp(-x[1])*abs(sin(t))
#   g(t) = x->g(x,t)
#   u0(x) = 0

#   T = Float64
#   reffe = ReferenceFE(lagrangian,T,order)
#   test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6])
#   trial = TransientTrialFESpace(test,g)
#   t0 = 0.
#   uh0μ = interpolate_everywhere(u0,trial(t0))
# end

test_g(x,t) = exp(-x[1])*abs(sin(t))
test_g(t) = x->test_g(x,t)
test_u0(x) = 0

test_trial = TransientTrialFESpace(test,test_g)
test_free_values = zero_free_values(test_trial)
# test_uh0 = interpolate_everywhere(test_u0,test_trial(t0))

f_test = test_trial(t0)
test_dirichlet_values = get_dirichlet_dof_values(f_test)
map(f_test.spaces,local_views(test_free_values),test_dirichlet_values) do V,fvec,dvec
  interpolate_everywhere!(test_u0,fvec,dvec,V)
end
test_a = test_free_values
# consistent!(test_a) |> wait
test_cache = map(reverse,test_a.cache)
test_vector_partition = partition(test_a)
test_buffer_snd = map(test_vector_partition,test_cache) do values,cache
  local_indices_snd = cache.local_indices_snd
  for (p,lid) in enumerate(local_indices_snd.data)
      cache.buffer_snd.data[p] = values[lid]
  end
  cache.buffer_snd
end

neighbors_snd,neighbors_rcv,test_buffer_rcv = map(test_cache) do cache
  cache.neighbors_snd,cache.neighbors_rcv,cache.buffer_rcv
end |> tuple_of_arrays
graph = ExchangeGraph(neighbors_snd,neighbors_rcv)
# test_t = exchange!(test_buffer_rcv,test_buffer_snd,graph)
test_T = eltype(eltype(test_buffer_snd))
# exchange_impl!(test_buffer_rcv,test_buffer_snd,graph,test_T)

function Base.:(==)(a::JaggedArray,b::JaggedArray)
  a.data == b.data && a.ptrs == b.ptrs
end

function check_ptjagged(a::PTJaggedArray,b::JaggedArray)
  c = JaggedArray(a.data[1],a.ptrs)
  c == b
end

check_ptjagged(cache[1].buffer_snd,test_cache[1].buffer_snd)
check_ptjagged(cache[2].buffer_snd,test_cache[2].buffer_snd)
check_ptjagged(cache[3].buffer_snd,test_cache[3].buffer_snd)
check_ptjagged(cache[4].buffer_snd,test_cache[4].buffer_snd)

function check_caches(
  ptcache::Vector{<:PTVectorAssemblyCache},
  cache::Vector{<:PartitionedArrays.VectorAssemblyCache})

  for i = eachindex(ptcache)
    ptci = ptcache[i]
    ci = cache[i]
    for f in propertynames(ptci)
      ptfi = getproperty(ptci,f)
      fi = getproperty(ci,f)
      msg = "AssertionError field $f"
      if isa(ptfi,PTJaggedArray)
        _fi = JaggedArray(ptfi.data[1],ptfi.ptrs)
        @assert _fi == fi msg
      else
        @assert ptfi == fi msg
      end
    end
  end
end

check_caches(cache,test_cache)
