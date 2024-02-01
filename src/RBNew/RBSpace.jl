abstract type RBSpace <: FESpace end

struct SingleFieldRBSpace{S<:SingleFieldFESpace,M<:AbstractMatrix} <: RBSpace
  space::S
  basis::M
end


function reduced_basis(
  rbinfo::RBInfo,
  feop::TransientParamFEOperator,
  s::TransientSnapshots)

  ϵ = rbinfo.ϵ
  nsnaps_state = rbinfo.nsnaps_state
  norm_matrix = get_norm_matrix(rbinfo,feop)
  return reduced_basis(s,norm_matrix;ϵ,nsnaps_state)
end

function reduced_basis(
  s::TransientSnapshots,
  norm_matrix;
  nsnaps_state=50,
  kwargs...)

  if size(s,1) < size(s,2)
    change_mode!(s)
  end
  sview = view(s,:,1:nsnaps_state)
  b1 = tpod(sview,norm_matrix;kwargs...)
  compressed_sview = b1'*sview
  change_mode!(compressed_sview)
  b2 = tpod(compressed_sview;kwargs...)
  if get_mode(s) == Mode1Axis()
    basis_space = b1
    basis_time = b2
  else
    basis_space = b2
    basis_time = b1
  end
  return basis_space,basis_time
end
