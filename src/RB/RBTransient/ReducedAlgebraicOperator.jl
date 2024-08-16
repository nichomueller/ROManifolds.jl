function RBSteady.reduce_operator(
  mdeim_style::SpaceOnlyMDEIM,
  b::TransientPODBasis,
  b_test::TransientPODBasis;
  kwargs...)

  bs = get_basis_space(b)
  bs_test = get_basis_space(b_test)
  bt_test = get_basis_time(b_test)

  T = eltype(bs_test)
  b̂st = Vector{Vector{Vector{T}}}(undef,RBSteady.num_reduced_space_dofs(b))

  b̂s = bs_test'*bs
  @inbounds for i = eachindex(b̂st)
    b̂si = b̂s[:,i]
    b̂st[i] = eachcol(kron(bt_test',b̂si))
  end

  return ReducedVectorOperator(mdeim_style,b̂st)
end

function RBSteady.reduce_operator(
  mdeim_style::SpaceOnlyMDEIM,
  b::TransientPODBasis,
  b_trial::TransientPODBasis,
  b_test::TransientPODBasis;
  kwargs...)

  bs = get_basis_space(b)
  bs_trial = get_basis_space(b_trial)
  bt_trial = get_basis_time(b_trial)
  bs_test = get_basis_space(b_test)
  bt_test = get_basis_time(b_test)

  T = promote_type(eltype(bs_trial),eltype(bs_test))
  b̂st = Vector{Vector{Matrix{T}}}(undef,RBSteady.num_reduced_space_dofs(b))

  b̂t = combine_basis_time(bt_trial,bt_test;kwargs...)

  @inbounds for i = eachindex(b̂st)
    b̂si = bs_test'*param_getindex(bs,i)*bs_trial
    b̂st[i] = map(k->kron(b̂t[k,:,:],b̂si),axes(b̂t,1))
  end

  return ReducedMatrixOperator(mdeim_style,b̂st)
end

function RBSteady.reduce_operator(
  mdeim_style::SpaceTimeMDEIM,
  b::TransientPODBasis,
  b_test::TransientPODBasis;
  kwargs...)

  bs = get_basis_space(b)
  bt = get_basis_time(b)
  bs_test = get_basis_space(b_test)
  bt_test = get_basis_time(b_test)

  T = eltype(bs_test)
  s = num_reduced_dofs(b_test),num_reduced_dofs(b)
  b̂st = Matrix{T}(undef,s)

  b̂s = bs_test'*bs
  b̂t = bt_test'*bt
  b̂st .= kron(b̂t,b̂s)

  return ReducedVectorOperator(mdeim_style,b̂st)
end

function RBSteady.reduce_operator(
  mdeim_style::SpaceTimeMDEIM,
  b::TransientPODBasis,
  b_trial::TransientPODBasis,
  b_test::TransientPODBasis;
  kwargs...)

  bs = get_basis_space(b)
  bt = get_basis_time(b)
  bs_trial = get_basis_space(b_trial)
  bt_trial = get_basis_time(b_trial)
  bs_test = get_basis_space(b_test)
  bt_test = get_basis_time(b_test)

  T = promote_type(eltype(bs_trial),eltype(bs_test))
  s = num_reduced_dofs(b_test),num_reduced_dofs(b),num_reduced_dofs(b_trial)
  b̂st = Array{T,3}(undef,s)

  b̂t = combine_basis_time(bt,bt_trial,bt_test;kwargs...)

  cache = zeros(T,num_space_dofs(b_test),RBSteady.num_reduced_space_dofs(b_trial))

  @inbounds for is = 1:RBSteady.num_reduced_space_dofs(b)
    Bis = param_getindex(bs,is)
    mul!(cache,Bis,bs_trial)
    b̂si = bs_test'*cache
    for it = 1:num_reduced_times(b)
      ist = (it-1)*RBSteady.num_reduced_space_dofs(b)+is
      b̂ti = b̂t[:,it,:]
      b̂st[:,ist,:] .= kron(b̂ti,b̂si)
    end
  end

  return ReducedMatrixOperator(mdeim_style,b̂st)
end

# TT interface

function RBSteady.reduce_operator(
  mdeim_style::SpaceTimeMDEIM,
  b::TransientTTSVDCores,
  b_test::TransientTTSVDCores;
  kwargs...)

  b̂st = compress_cores(b,b_test)
  return ReducedVectorOperator(mdeim_style,b̂st)
end

function RBSteady.reduce_operator(
  mdeim_style::SpaceTimeMDEIM,
  b::TransientTTSVDCores,
  b_trial::TransientTTSVDCores,
  b_test::TransientTTSVDCores;
  kwargs...)

  b̂st = compress_cores(b,b_trial,b_test;kwargs...)
  return ReducedMatrixOperator(mdeim_style,b̂st)
end

function RBSteady.compress_core(a::AbstractArray{T,3},btrial::AbstractArray{S,3},btest::AbstractArray{S,3};
  combine=(x,y)->x) where {T,S}

  TS = promote_type(T,S)
  ra_prev,ra = size(a,1),size(a,3)
  rU_prev,rU = size(btrial,1),size(btrial,3)
  rV_prev,rV = size(btest,1),size(btest,3)
  bab = zeros(TS,rV_prev,ra_prev,rU_prev,rV,ra,rU)
  bab_shift = zeros(TS,rV_prev,ra_prev,rU_prev,rV,ra,rU)
  w = zeros(TS,size(a,2))
  w_shift = zeros(TS,size(a,2)-1)
  for ibU1 = 1:rU_prev
    @inbounds bU′ = btrial[ibU1,:,:]
    for ia1 = 1:ra_prev
      @inbounds a′ = a[ia1,:,:]
      for ibU3 = 1:rU
        @inbounds bU′′ = bU′[:,ibU3]
        for ia3 = 1:ra
          @inbounds a′′ = a′[:,ia3]
          w .= a′′.*bU′′
          w_shift .= a′′[2:end].*bU′′[1:end-1]
          for ibV1 = 1:rV_prev
            @inbounds bV′ = btest[ibV1,:,:]
            for ibV3 = 1:rV
              @inbounds bV′′ = bV′[:,ibV3]
              @inbounds bab[ibV1,ia1,ibU1,ibV3,ia3,ibU3] = dot(bV′′,w)
              @inbounds bab_shift[ibV1,ia1,ibU1,ibV3,ia3,ibU3] = dot(bV′′[2:end],w_shift)
            end
          end
        end
      end
    end
  end
  return combine(bab,bab_shift)
end

function combine_basis_time(B::AbstractMatrix,C::AbstractMatrix;combine=(x,y)->x)
  time_ndofs = size(C,1)
  nt_row = size(C,2)
  nt_col = size(B,2)

  T = eltype(C)
  bt_proj = zeros(T,time_ndofs,nt_row,nt_col)
  bt_proj_shift = copy(bt_proj)
  @inbounds for jt = 1:nt_col, it = 1:nt_row
    bt_proj[:,it,jt] .= C[:,it].*B[:,jt]
    bt_proj_shift[2:end,it,jt] .= C[2:end,it].*B[1:end-1,jt]
  end

  combine(bt_proj,bt_proj_shift)
end

function combine_basis_time(A::AbstractMatrix,B::AbstractMatrix,C::AbstractMatrix;combine=(x,y)->x)
  nt = size(A,2)
  nt_row = size(C,2)
  nt_col = size(B,2)

  T = promote_type(eltype(B),eltype(C))
  bt_proj = zeros(T,nt_row,nt,nt_col)
  bt_proj_shift = copy(bt_proj)
  @inbounds for q = 1:nt, jt = 1:nt_col, it = 1:nt_row
    bt_proj[it,q,jt] = sum(C[:,it].*A[:,q].*B[:,jt])
    bt_proj_shift[it,q,jt] = sum(C[2:end,it].*A[2:end,q].*B[1:end-1,jt])
  end

  combine(bt_proj,bt_proj_shift)
end

function Base.:*(a::ReducedVectorOperator{SpaceOnlyMDEIM,Vector{<:Vector}},b::AbstractMatrix)
  return sum([a.basis[q]*b[q,:]' for q = eachindex(a.basis)])
end

function Base.:*(a::ReducedMatrixOperator{SpaceOnlyMDEIM,Vector{<:Vector}},b::AbstractMatrix)
  return sum([a.basis[q][k]*b[q,k] for q = eachindex(a.basis) for k = eachindex(a.basis[q])])
end
