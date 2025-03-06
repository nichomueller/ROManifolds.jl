function RBSteady.reduced_operator(
  solver::RBSolver,
  feop::TransientParamFEOperator,
  args...;
  kwargs...)

  fesnaps,festats = solution_snapshots(solver,feop,args...;kwargs...)
  reduced_operator(solver,feop,fesnaps)
end

function RBSteady.reduced_operator(
  solver::RBSolver,
  feop::TransientParamFEOperator,
  s::AbstractSnapshots,
  args...)

  red_trial,red_test = reduced_spaces(solver,feop,s)
  odeop = get_algebraic_operator(feop)
  reduced_operator(solver,odeop,red_trial,red_test,s)
end

function RBSteady.reduced_operator(
  solver::RBSolver,
  odeop::ODEParamOperator,
  red_trial::RBSpace,
  red_test::RBSpace,
  s::AbstractSnapshots)

  red_lhs,red_rhs = reduced_weak_form(solver,odeop,red_trial,red_test,s)
  trians_rhs = get_domains(red_rhs)
  trians_lhs = map(get_domains,red_lhs)
  odeop′ = change_domains(odeop,trians_rhs,trians_lhs)
  GenericRBOperator(odeop′,red_trial,red_test,red_lhs,red_rhs)
end

function RBSteady.reduced_operator(
  solver::RBSolver,
  odeop::ODEParamOperator{LinearNonlinearParamODE},
  red_trial::RBSpace,
  red_test::RBSpace,
  s::AbstractSnapshots)

  red_op_lin = reduced_operator(solver,get_linear_operator(odeop),red_trial,red_test,s)
  red_op_nlin = reduced_operator(solver,get_nonlinear_operator(odeop),red_trial,red_test,s)
  LinearNonlinearRBOperator(red_op_lin,red_op_nlin)
end

const TransientRBOperator{O} = GenericRBOperator{TupOfAffineContribution,O}

function Algebra.allocate_residual(
  op::TransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  paramcache)

  allocate_hypred_cache(op.rhs,r)
end

function Algebra.allocate_jacobian(
  op::TransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  paramcache)

  allocate_hypred_cache(op.lhs,r)
end

function Algebra.residual!(
  b::HRParamArray,
  op::TransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  paramcache)

  hr_residual!(b,op,r,us,paramcache)
  inv_project!(b,op.rhs,feb)
end

function Algebra.jacobian!(
  A::HRParamArray,
  op::TransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  ws::Tuple{Vararg{Real}},
  paramcache)

  feA = hr_jacobian!(A,op,r,u,paramcache)
  inv_project!(A,op.lhs,feA)
end

# function RBSteady.hr_residual!(
#   b,
#   op::TransientRBOperator,
#   r::TransientRealization,
#   us::Tuple{Vararg{AbstractParamVector}},
#   paramcache)

#   red_params = 1:num_params(r)
#   red_times = union_indices_time(op.rhs)
#   red_pt_indices = range_2d(red_params,red_times,num_params(r))
#   red_r = r[red_params,red_times]

#   red_b,red_us,red_odeopcache = select_fe_quantities_at_indices(b,us,paramcache,vec(red_pt_indices))
#   residual!(red_b,op.op,red_r,red_us,red_odeopcache)
#   RBSteady.select_at_indices(red_b,op.rhs,red_pt_indices)
# end

# function RBSteady.hr_jacobian!(
#   A,
#   op::TransientRBOperator,
#   r::TransientRealization,
#   us::Tuple{Vararg{AbstractParamVector}},
#   ws::Tuple{Vararg{Real}},
#   paramcache)

#   red_params = 1:num_params(r)
#   red_times = union_indices_time(op.lhs)
#   red_pt_indices = range_2d(red_params,red_times,num_params(r))
#   red_r = r[red_params,red_times]

#   red_A,red_us,red_odeopcache = select_fe_quantities_at_indices(A,us,paramcache,vec(red_pt_indices))
#   jacobian!(red_A,op.op,red_r,red_us,ws,red_odeopcache)
#   map(red_A,op.lhs) do red_A,lhs
#     RBSteady.select_at_indices(red_A,lhs,red_pt_indices)
#   end
# end

# """
#     struct LinearNonlinearTransientRBOperator <: TransientRBOperator{LinearNonlinearParamODE}
#       op_linear::TransientRBOperator{LinearParamODE}
#       op_nonlinear::TransientRBOperator{NonlinearParamODE}
#     end

# Extends the concept of [`TransientRBOperator`](@ref) to accommodate the linear/nonlinear
# splitting of terms in nonlinear applications
# """
# struct LinearNonlinearTransientRBOperator <: TransientRBOperator{LinearNonlinearParamODE}
#   op_linear::TransientRBOperator{LinearParamODE}
#   op_nonlinear::TransientRBOperator{NonlinearParamODE}
# end

# ParamAlgebra.get_linear_operator(op::LinearNonlinearTransientRBOperator) = op.op_linear
# ParamAlgebra.get_nonlinear_operator(op::LinearNonlinearTransientRBOperator) = op.op_nonlinear

# function FESpaces.get_test(op::LinearNonlinearTransientRBOperator)
#   @check get_test(op.op_linear) === get_test(op.op_nonlinear)
#   get_test(op.op_nonlinear)
# end

# function FESpaces.get_trial(op::LinearNonlinearTransientRBOperator)
#   @check get_trial(op.op_linear) === get_trial(op.op_nonlinear)
#   get_trial(op.op_nonlinear)
# end

# function RBSteady.get_fe_trial(op::LinearNonlinearTransientRBOperator)
#   @check RBSteady.get_fe_trial(op.op_linear) === get_fe_trial(op.op_nonlinear)
#   RBSteady.get_fe_trial(op.op_nonlinear)
# end

# function RBSteady.get_fe_test(op::LinearNonlinearTransientRBOperator)
#   @check RBSteady.get_fe_test(op.op_linear) === RBSteady.get_fe_test(op.op_nonlinear)
#   RBSteady.get_fe_test(op.op_nonlinear)
# end

# function Algebra.allocate_residual(
#   op::LinearNonlinearTransientRBOperator,
#   r::TransientRealization,
#   us::Tuple{Vararg{AbstractParamVector}},
#   rbcache::LinearNonlinearRBCache)

#   allocate_residual(get_nonlinear_operator(op),r,us,rbcache.rbcache)
# end

# function Algebra.allocate_jacobian(
#   op::LinearNonlinearTransientRBOperator,
#   r::TransientRealization,
#   us::Tuple{Vararg{AbstractParamVector}},
#   rbcache::LinearNonlinearRBCache)

#   allocate_jacobian(get_nonlinear_operator(op),r,us,rbcache.rbcache)
# end

# function Algebra.residual!(
#   cache,
#   op::LinearNonlinearTransientRBOperator,
#   r::TransientRealization,
#   us::Tuple{Vararg{AbstractParamVector}},
#   rbcache::LinearNonlinearRBCache)

#   nlop = get_nonlinear_operator(op)
#   A_lin = rbcache.A
#   b_lin = rbcache.b
#   rbcache_nlin = rbcache.rbcache

#   b_nlin = residual!(cache,nlop,r,us,rbcache_nlin)
#   axpy!(1.0,b_lin,b_nlin)
#   mul!(b_nlin,A_lin,us[end],true,true)

#   return b_nlin
# end

# function Algebra.jacobian!(
#   cache,
#   op::LinearNonlinearTransientRBOperator,
#   r::TransientRealization,
#   us::Tuple{Vararg{AbstractParamVector}},
#   ws::Tuple{Vararg{Real}},
#   rbcache::LinearNonlinearRBCache)

#   nlop = get_nonlinear_operator(op)
#   A_lin = rbcache.A
#   rbcache_nlin = rbcache.rbcache

#   A_nlin = jacobian!(cache,nlop,r,us,ws,rbcache_nlin)
#   axpy!(1.0,A_lin,A_nlin)

#   return A_nlin
# end

# function Algebra.solve!(
#   x̂::AbstractParamVector,
#   solver::RBSolver,
#   op::TransientRBOperator,
#   r::TransientRealization,
#   x::AbstractParamVector,
#   x0::AbstractParamVector)

#   fesolver = get_fe_solver(solver)
#   rbcache = allocate_rbcache(fesolver,op,r,x)

#   t = @timed solve!(x̂,fesolver,op,r,x,x0,rbcache)
#   stats = CostTracker(t,nruns=num_params(r),name="RB solver")

#   return x̂,stats
# end

# # cache utils

# function select_fe_space_at_indices(fs::FESpace,indices)
#   @notimplemented
# end

# function select_fe_space_at_indices(fs::TrivialParamFESpace,indices)
#   TrivialParamFESpace(fs.space,length(indices))
# end

# function select_fe_space_at_indices(fs::TrialParamFESpace,indices)
#   dvi = ConsecutiveParamArray(view(fs.dirichlet_values.data,:,indices))
#   TrialParamFESpace(dvi,fs.space)
# end

# function select_slvrcache_at_indices(b::ConsecutiveParamArray,indices)
#   ConsecutiveParamArray(view(b.data,:,indices))
# end

# function select_slvrcache_at_indices(A::ConsecutiveParamSparseMatrixCSC,indices)
#   ConsecutiveParamSparseMatrixCSC(A.m,A.n,A.colptr,A.rowval,A.data[:,indices])
# end

# function select_slvrcache_at_indices(A::BlockParamArray,indices)
#   map(a -> select_slvrcache_at_indices(a,indices),blocks(A)) |> mortar
# end

# function select_slvrcache_at_indices(cache::ArrayContribution,indices)
#   contribution(cache.trians) do trian
#     select_slvrcache_at_indices(cache[trian],indices)
#   end
# end

# function select_slvrcache_at_indices(cache::TupOfArrayContribution,indices)
#   red_cache = ()
#   for c in cache
#     red_cache = (red_cache...,select_slvrcache_at_indices(c,indices))
#   end
#   return red_cache
# end

# function select_evalcache_at_indices(us::Tuple{Vararg{ConsecutiveParamVector}},paramcache,indices)
#   @unpack trial,ptrial = paramcache
#   new_xhF = ()
#   new_trial = ()
#   for i = eachindex(trial)
#     new_trial = (new_trial...,select_fe_space_at_indices(trial[i],indices))
#     new_XhF_i = ConsecutiveParamArray(view(us[i].data,:,indices))
#     new_xhF = (new_xhF...,new_XhF_i)
#   end
#   new_odeopcache = ParamCache(new_trial,ptrial)
#   return new_xhF,new_odeopcache
# end

# function select_evalcache_at_indices(us::Tuple{Vararg{BlockConsecutiveParamVector}},paramcache,indices)
#   @unpack trial,ptrial = paramcache
#   new_xhF = ()
#   new_trial = ()
#   for i = eachindex(trial)
#     spacei = trial[i]
#     VT = spacei.vector_type
#     style = spacei.multi_field_style
#     spacesi = [select_fe_space_at_indices(spaceij,indices) for spaceij in spacei]
#     new_trial = (new_trial...,MultiFieldFESpace(VT,spacesi,style))
#     new_XhF_i = mortar([ConsecutiveParamArray(view(us_i.data,:,indices)) for us_i in blocks(us[i])])
#     new_xhF = (new_xhF...,new_XhF_i)
#   end
#   new_odeopcache = ParamCache(new_trial,ptrial)
#   return new_xhF,new_odeopcache
# end

# function select_fe_quantities_at_indices(cache,us,paramcache,indices)
#   # returns the cache in the appropriate time-parameter locations
#   red_cache = select_slvrcache_at_indices(cache,indices)
#   # does the same with the stage variable `us` and the ode cache `paramcache`
#   red_us,red_odeopcache = select_evalcache_at_indices(us,paramcache,indices)
#   return red_cache,red_us,red_odeopcache
# end

# get_entry(s::ConsecutiveParamVector,is,ipt) = get_all_data(s)[is,ipt]
# get_entry(s::ParamSparseMatrix,is,ipt) = param_getindex(s,ipt)[is]

# function RBSteady.select_at_indices(
#   ::TransientHyperReduction,
#   a::AbstractParamArray,
#   ids_space,ids_time,ids_param)

#   @check length(ids_space) == length(ids_time)
#   entries = zeros(eltype2(a),length(ids_space),length(ids_param))
#   @inbounds for ip = 1:length(ids_param)
#     for (i,(is,it)) in enumerate(zip(ids_space,ids_time))
#       ipt = ip+(it-1)*length(ids_param)
#       v = get_entry(a,is,ipt)
#       entries[i,ip] = v
#     end
#   end
#   return ConsecutiveParamArray(entries)
# end

# function RBSteady.select_at_indices(
#   ::TransientHyperReduction{<:TransientReduction},
#   a::AbstractParamArray,
#   ids_space,ids_time,ids_param)

#   entries = zeros(eltype2(a),length(ids_space),length(ids_time),length(ids_param))
#   @inbounds for ip = 1:length(ids_param)
#     for (i,it) in enumerate(ids_time)
#       ipt = ip+(it-1)*length(ids_param)
#       v = get_entry(a,ids_space,ipt)
#       @views entries[:,i,ip] = v
#     end
#   end
#   return ConsecutiveParamArray(entries)
# end

# function RBSteady.select_at_indices(s::AbstractArray,a::TransientHyperReduction,indices::Range2D)
#   ids_space = get_indices_space(a)
#   ids_param = indices.axis1
#   common_ids_time = indices.axis2
#   domain_time = get_integration_domain_time(a)
#   ids_time = RBSteady.ordered_common_locations(domain_time,common_ids_time)
#   RBSteady.select_at_indices(a,s,ids_space,ids_time,ids_param)
# end

# function ordered_common_locations(i::VectorDomain,uindices::AbstractVector)::Vector{Int}
#   filter(!isnothing,indexin(i,uindices))
# end

# function RBSteady.select_at_indices(
#   s::ArrayContribution,a::AffineContribution,indices)
#   contribution(s.trians) do trian
#     RBSteady.select_at_indices(s[trian],a[trian],indices)
#   end
# end
