function Gridap.FESpaces.allocate_vector(
  a::SparseMatrixAssembler,
  vecdata,
  filter)

  r,d = _filter_vecdata(a,vecdata,filter)
  vec = allocate_vector(a,d)
  vec[r]
end

function Gridap.FESpaces.assemble_vector(
  a::SparseMatrixAssembler,
  vecdata,
  filter)

  r,d = _filter_vecdata(a,vecdata,filter)
  vec = assemble_vector(a,d)
  vec[r]
end

function Gridap.FESpaces.allocate_matrix(
  a::SparseMatrixAssembler,
  matdata,
  filter)

  r,c,d = _filter_matdata(a,matdata,filter)
  mat = allocate_matrix(a,d)
  mat[r,c]
end

function Gridap.FESpaces.assemble_matrix(
  a::SparseMatrixAssembler,
  matdata,
  filter)

  r,c,d = _filter_matdata(a,matdata,filter)
  mat = assemble_matrix(a,d)
  mat[r,c]
end

function Gridap.FESpaces.assemble_matrix_and_vector(
  a::SparseMatrixAssembler,
  data,
  filters)

  matfilter,vecfilter = filters
  matvecdata,matdata,vecdata = data
  _,_,dmv = _filter_matdata(a,matvecdata,matfilter)
  rm,cm,dm = _filter_matdata(a,matdata,matfilter)
  rv,dv = _filter_vecdata(a,vecdata,vecfilter)
  mat,vec = assemble_matrix_and_vector(a,(dmv,dm,dv))
  mat[rm,cm],vec[rv]
end

function _filter_vecdata(
  a::SparseMatrixAssembler,
  vecdata::Tuple{Vararg{Any}},
  filter)

  vals,rowids, = vecdata
  r_filter, = filter
  r = _idx_in_block(get_rows(a),r_filter)
  d = _filter_data(vals,filter),_filter_data(rowids,r_filter)
  r,d
end

function _filter_matdata(
  a::SparseMatrixAssembler,
  matdata::Tuple{Vararg{Any}},
  filter)

  vals,rowids,colids = matdata
  r_filter,c_filter = filter
  r = _idx_in_block(get_rows(a),r_filter)
  c = _idx_in_block(get_cols(a),c_filter)
  d = _filter_data(vals,filter),_filter_data(rowids,r_filter),_filter_data(colids,c_filter)
  r,c,d
end

function _idx_in_block(ndofs::Vector{Int},filter::Int)
  @assert filter == 1
  first(ndofs):last(ndofs)
end

function _idx_in_block(ndofs::BlockedUnitRange,filter::Int)
  nd = [0,ndofs.lasts...]
  [idx1+1:idx2 for (idx1,idx2) in zip(nd[1:end-1],nd[2:end])][filter]
end

_filter_data(data,args...) = data

function _filter_data(data::Vector{Any},filter)
  [_filter_data(d,filter) for d = data]
end

function _filter_data(data::LazyArray,filter)
  lazy_map(d->_filter_data(d,filter),data)
end

function _filter_data(data::ArrayBlock,filter)
  data[filter...]
end

function _filter_data(
  data::Tuple{MatrixBlock{Matrix{Float}},VectorBlock{Vector{Float}}},
  filter::NTuple{2,Int})

  mdata,vdata = data
  r_filter,c_filter = filter
  mdata[r_filter,c_filter],vdata[r_filter]
end

# function Gridap.FESpaces.collect_cell_vector(
#   test::FESpace,
#   liform::Function,
#   params::Table)

#   veccontribs = pmap(liform,params)
#   collect_cell_vector(test,veccontribs)
# end

# function Gridap.FESpaces.collect_cell_vector(
#   test::FESpace,
#   liform::Function,
#   params::Table,
#   times::Vector)

#   veccontribs = pmap(μ -> map(t -> liform(μ,t),times),params)
#   collect_cell_vector(test,veccontribs)
# end

# function Gridap.FESpaces.collect_cell_vector(
#   test::FESpace,
#   veccontribs::Vector{DomainContribution})

#   data = pmap(v -> collect_cell_matrix_and_vector(test,v),veccontribs)
#   pmap(d->getindex(d,2),data)
# end

# function Gridap.FESpaces.collect_cell_matrix(
#   trial::ParamTrialFESpace,
#   test::FESpace,
#   biform::Function,
#   params::Table)

#   trials = pmap(trial,params)
#   matcontribs = pmap(biform,params)
#   collect_cell_matrix(trials,test,matcontribs)
# end

# function Gridap.FESpaces.collect_cell_matrix(
#   trial::ParamTransientTrialFESpace,
#   test::FESpace,
#   biform::Function,
#   params::Table,
#   times::Vector)

#   trials = pmap(μ -> map(t -> trial(μ,t),times),params)
#   matcontribs = pmap(μ -> map(t -> biform(μ,t),times),params)
#   collect_cell_matrix(trials,test,matcontribs)
# end

# function Gridap.FESpaces.collect_cell_matrix(
#   trials::Vector{TrialFESpace},
#   test::FESpace,
#   matcontribs::Vector{DomainContribution})

#   data = pmap((tr,m) -> collect_cell_matrix(tr,test,m),trials,matcontribs)
#   pmap(d->getindex(d,1),data)
# end

# MDEIM snapshots generation interface

function assemble_residual(
  odeop::ParamFEOperator,
  params::Table,
  uh::Vector{T},
  filter) where {T<:AbstractArray}

  b = allocate_residual(odeop,first(u),nothing)
  trial = get_trial(op)
  test = get_test(op)
  dv = get_fe_basis(test)
  u(μ) = pmap(x -> EvaluationFunction(trial(μ),x),uh)
  vecdatum(μ) = collect_cell_vector(
    test,
    op.res(μ,u(μ),dv),
    params,
    times,
    uh)
  vecdata = pmap(μ -> map(t -> vecdatum(μ),times),params,uh)
  pmap(d -> assemble_residual!(b,op.assem,d),vecdata...)
end

function assemble_jacobian(
  op::ParamFEOperator,
  params::Table,
  uh::Vector{T},
  filter) where {T<:AbstractArray}

  A = allocate_jacobian(op,first(uh),nothing)
  trial = get_trial(op)
  test = get_test(op)
  dv = get_fe_basis(test)
  du = get_trial_fe_basis(trial(nothing))
  u(μ) = pmap(x -> EvaluationFunction(trial(μ),x),uh)
  matdatum(μ) = collect_cell_matrix(trial(μ),test,op.jac(μ,u(μ),dv,du))
  vecdata = pmap(μ -> map(t -> matdatum(μ),times),params,uh)
  pmap(d -> assemble_jacobian!(A,op.assem,d),vecdata...)
end





# function Gridap.FESpaces.collect_cell_matrix_and_vector(
#   trial::ParamTrialFESpace,
#   test::FESpace,
#   biform::Function,
#   liform::Function,
#   params::Table)

#   trials = pmap(trial,params)
#   matcontribs = pmap(biform,params)
#   veccontribs = pmap(liform,params)
#   collect_cell_matrix_and_vector(trials,test,matcontribs,veccontribs)
# end

# function Gridap.FESpaces.collect_cell_matrix_and_vector(
#   trial::ParamTransientTrialFESpace,
#   test::FESpace,
#   biform::Function,
#   liform::Function,
#   params::Table,
#   times::Vector)

#   trials = pmap(μ -> map(t -> trial(μ,t),times),params)
#   matcontribs = pmap(μ -> map(t -> biform(μ,t),times),params)
#   veccontribs = pmap(μ -> map(t -> liform(μ,t),times),params)
#   collect_cell_matrix_and_vector(trials,test,matcontribs,veccontribs)
# end

# function Gridap.FESpaces.collect_cell_matrix_and_vector(
#   trials::Vector{TrialFESpace},
#   test::FESpace,
#   matcontribs::Vector{DomainContribution},
#   veccontribs::Vector{DomainContribution})

#   data = pmap((t,m,v) -> collect_cell_matrix_and_vector(t,test,m,v),
#     trials,matcontribs,veccontribs)
#   matvecdata = pmap(d->getindex(d,1),data)
#   matdata = pmap(d->getindex(d,2),data)
#   vecdata = pmap(d->getindex(d,3),data)
#   matvecdata,matdata,vecdata
# end

for T in (:ParamMultiFieldTrialFESpace,:ParamTransientMultiFieldTrialFESpace)

  @eval begin
    function get_snapshots(
      trial::T,
      test::FESpace,
      biform::Function,
      liform::Function,
      args...)

      am = SparseMatrixAssembler(trial,test)
      av = SparseMatrixAssembler(test,test)
      matvecdata,matdata,vecdata =
        collect_cell_matrix_and_vector(trial,test,biform,liform,args...)
      nfields = test.nfields
      for r_filter = 1:nfields, c_filter = 1:nfields
        vecs = allocate_vector()
      end
    end

  end

end
