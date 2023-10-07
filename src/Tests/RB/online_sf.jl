ℓinf(x) = maximum(abs.(x))

times = get_times(fesolver)
nparams = 10
ntimes = length(times)
θ = fesolver.θ
i = 1
combine_projections = (x,y) -> i == 1 ? θ*x+(1-θ)*y : x-y
jacs,trian = collect_jacobians_for_trian(fesolver,feop,sols[1:nparams],params[1:nparams],times;i)

A_fun(μ,t) = assemble_matrix((u,v)->∫(a(μ,t)*∇(v)⋅∇(u))dΩ,trial(μ,t),test)
boh = Vector{Float}[]
for μ in params[1:nparams]
  for t in times
    innz,Annz = findnz(A_fun(μ,t)[:])
    push!(boh,Annz)
  end
end
snapsA = hcat(boh...)

# ad_jac_i = compress_component(info,feop,jacs,trian,times,rbspace,rbspace;combine_projections)
nzm = jacs[1]
basis_space,basis_time = compress(nzm;ϵ=info.ϵ)
m2 = change_mode(nzm)
@check all([snapsA[:,n] ≈ nzm.nonzero_val[:,n] for n = 1:ntimes*nparams])
@check isapprox(m2,hcat([nzm.nonzero_val[:,(i-1)*ntimes+1:i*ntimes]' for i = 1:nparams]...))

proj_bs,proj_bt = project_space_time(basis_space,basis_time,rbspace,rbspace;combine_projections)
irow,icol = vec_to_mat_idx(basis_space.nonzero_idx,basis_space.nrows)
ncols = maximum(icol)
for (n,nzv) in enumerate(eachcol(basis_space))
  B = sparse(irow,icol,nzv,basis_space.nrows,ncols)
  Bred = rbspace.basis_space'*B*rbspace.basis_space
  Bred ≈ proj_bs[n]
end

bs = basis_space.nonzero_val
@check maximum(abs.(nzm.nonzero_val - bs*bs'*nzm.nonzero_val)) ≤ 10*ϵ

# basis_space = POD(snapsA;ϵ)
interp_idx_space = get_interpolation_idx(basis_space)
interp_idx_time = get_interpolation_idx(basis_time)
entire_interp_idx_space = recast_idx(nzm,interp_idx_space)
entire_interp_idx_rows,_ = vec_to_mat_idx(entire_interp_idx_space,nzm.nrows)

Arand = A_fun(params[end],pi)
nnzi,_ = findnz(Arand[:])
@check all([x ∈ nnzi for x in entire_interp_idx_space])

interp_bs = basis_space[interp_idx_space,:]
lu_interp = lu(interp_bs)
x = rand(size(interp_bs,2))
v = rand(size(interp_bs,2))
ldiv!(v,lu_interp,x)
@check v ≈ interp_bs \ x

trian = Ω
cell_dof_ids = get_cell_dof_ids(feop.test,trian)
red_integr_cells = find_cells(entire_interp_idx_rows,cell_dof_ids)
red_trian = view(trian,red_integr_cells)
red_meas = get_measure(feop,red_trian)
red_times = st_mdeim ? times[interp_idx_time] : times
integr_domain = RBIntegrationDomain(red_meas,red_times,entire_interp_idx_space)

p,t = realization(feop),3/2*dt
Ared = assemble_matrix((u,v)->∫(a(p,t)*∇(v)⋅∇(u))red_meas,trial(p,t),test)
Annz_pt = Ared[:][entire_interp_idx_space]
x = rand(size(interp_bs,2))
coeffA = ldiv!(x,lu_interp,Annz_pt)

_,Annz_pt_ok = findnz(A_fun(p,t)[:])
coeffA_ok = bs' * Annz_pt_ok

ℓinf(coeffA - coeffA_ok)

# try comparison with old MDEIM

# function mdeim_idx(M::Matrix{Float})
#   n = size(M)[2]
#   idx = Int[]
#   append!(idx,Int(argmax(abs.(M[:,1]))))

#   @inbounds for i = 2:n
#     res = (M[:,i] - M[:,1:i-1] *
#       (M[idx[1:i-1],1:i-1] \ M[idx[1:i-1],i]))
#     append!(idx,Int(argmax(abs.(res))))
#   end

#   unique(idx)
# end

# recast_in_full_dim(idx_tmp::Vector{Int},findnz_map::Vector{Int}) = findnz_map[idx_tmp]

# function rb_space_projection(
#   mat::Matrix{Float},
#   findnz_map::Vector{Int},
#   rbspace_row::RBSpace,
#   rbspace_col::RBSpace)

#   sparse_basis_space = _sparsevec(mat,findnz_map)
#   brow = rbspace_row.basis_space
#   bcol = rbspace_col.basis_space

#   Qs = length(sparse_basis_space)
#   Ns = size(bcol,1)
#   red_basis_space = zeros(size(brow,2)*size(bcol,2),Qs)
#   for q = 1:Qs
#     smat = sparsevec_to_sparsemat(sparse_basis_space[q],Ns)
#     red_basis_space[:,q] = (brow'*smat*bcol)[:]
#   end

#   red_basis_space
# end

# function _sparsevec(M::Matrix{T},findnz_map::Vector{Int}) where T
#   sparse_vblocks = SparseVector{T}[]
#   for j = axes(M,2)
#     push!(sparse_vblocks,sparsevec(findnz_map,M[:,j],maximum(findnz_map)))
#   end

#   sparse_vblocks
# end

# function sparsevec_to_sparsemat(svec::SparseVector,Nc::Int)
#   ij,v = findnz(svec)
#   i,j = from_vec_to_mat_idx(ij,Nc)
#   sparse(i,j,v,maximum(i),Nc)
# end

# function get_red_measure(
#   feop,
#   idx::Vector{Int},
#   trian::Triangulation)

#   el = find_mesh_elements(feop,idx,trian)
#   red_trian = view(trian,el)
#   Measure(red_trian,2)
# end

# function find_mesh_elements(
#   feop,
#   idx_tmp::Vector{Int},
#   trian::Triangulation)

#   idx = recast_in_mat_form(idx_tmp,feop.test.nfree)
#   connectivity = get_cell_dof_ids(feop.test,trian)

#   el = Int[]
#   for i = eachindex(idx)
#     for j = axes(connectivity,1)
#       if idx[i] in abs.(connectivity[j])
#         append!(el,j)
#       end
#     end
#   end

#   unique(el)
# end

# function from_vec_to_mat_idx(idx::Vector{Int},Ns::Int)
#   col_idx = 1 .+ Int.(floor.((idx.-1)/Ns))
#   findnz_map = idx - (col_idx.-1)*Ns
#   findnz_map,col_idx
# end

# function recast_in_mat_form(idx_tmp::Vector{Int},Ns::Int)
#   idx_space,_ = from_vec_to_mat_idx(idx_tmp,Ns)
#   idx_space
# end


# function POD(S::AbstractMatrix;ϵ=1e-4)
#   U,Σ,_ = svd(S)
#   energies = cumsum(Σ.^2)
#   n = findall(x->x ≥ (1-ϵ^2)*energies[end],energies)[1]
#   err = sqrt(1-energies[n]/energies[end])
#   printstyled("Basis number obtained via POD is $n, projection error ≤ $err\n";
#     color=:blue)

#   U[:,1:n]
# end

# findnz_map = nnzi
# _bs_proj = rb_space_projection(basis_space.nonzero_val,findnz_map,rbspace,rbspace)
# __idx = mdeim_idx(basis_space.nonzero_val)
# red_lu_factors = lu(basis_space.nonzero_val[__idx,:])
# _idx = recast_in_full_dim(__idx,findnz_map)
# _red_meas = get_red_measure(feop,_idx,Ω)

# @assert basis_space ≈ POD(snapsA;ϵ)
# @assert __idx == interp_idx_space
# @assert _idx == entire_interp_idx_space
# @assert red_lu_factors.L == lu_interp.L && red_lu_factors.U == lu_interp.U

# @assert all([proj_bs[i] ≈ reshape(_bs_proj[:,i],16,16) for i = eachindex(proj_bs)])


# WORKS FOR A VECTOR
times = get_times(fesolver)
nparams = 10
ntimes = length(times)

H_fun(μ,t) = assemble_vector(v->∫(g(μ,t)*v)dΩ,test)

v = Vector{Float}[]
for μ in params[1:nparams]
  for t in times
    push!(v,H_fun(μ,t))
  end
end
_,snapsH = compress_array(hcat(v...))

basis_space = tpod(snapsH;ϵ=info.ϵ)
proj_bs = [rbspace.basis_space'*b for b in eachcol(basis_space)]

bs = basis_space
@check maximum(abs.(snapsH - bs*bs'*snapsH)) ≤ 10*ϵ

interp_idx_space = get_interpolation_idx(basis_space)
entire_interp_idx_space = interp_idx_space
entire_interp_idx_rows,_ = vec_to_mat_idx(entire_interp_idx_space,nzm.nrows)

interp_bs = basis_space[interp_idx_space,:]
lu_interp = lu(interp_bs)
x = rand(size(interp_bs,2))
v = rand(size(interp_bs,2))
ldiv!(v,lu_interp,x)
@check v ≈ interp_bs \ x

trian = Ω
cell_dof_ids = get_cell_dof_ids(feop.test,trian)
red_integr_cells = find_cells(entire_interp_idx_rows,cell_dof_ids)
red_trian = view(trian,red_integr_cells)
red_meas = get_measure(feop,red_trian)
red_times = st_mdeim ? times[interp_idx_time] : times
integr_domain = RBIntegrationDomain(red_meas,red_times,entire_interp_idx_space)

p,t = realization(feop),pi
Hred = assemble_vector(v->∫(g(p,t)*v)red_meas,test)
Hnnz_pt = Hred[entire_interp_idx_space]
x = rand(size(interp_bs,2))
coeffH = ldiv!(x,lu_interp,Hnnz_pt)

Hnnz_pt_ok = H_fun(p,t)
coeffH_ok = basis_space' * Hnnz_pt_ok

ℓinf(coeffH - coeffH_ok)

















# TRY WITH JACOBIAN
i = 1
trian = Ω
rbjact = rbjac[i][trian]
meas = dΩ
cache = jac_cache[1]

jcache,scache... = cache

times = get_times(fesolver)
ndofs_row = num_free_dofs(feop.test)
ndofs_col = num_free_dofs(get_trial(feop)(nothing,nothing))
setsize!(jcache,(ndofs_row,ndofs_col))

red_idx = rbjact.integration_domain.idx
red_times = rbjact.integration_domain.times
red_meas = rbjact.integration_domain.meas

A = get_array(jcache;len=length(red_times)*length(params_test))
sols_test = get_solutions_at_times(sols_test,fesolver,red_times)

Afull = copy(A)
full_idx = findnz(Afull[1][:])[1]
Jac = collect_jacobians_for_idx!(A,fesolver,sols_test,params_test,red_times,red_idx,red_meas;i)
Jac_full = collect_jacobians_for_idx!(Afull,fesolver,sols_test,params_test,red_times,full_idx,meas;i)
Jac_offline,_ = collect_jacobians_for_trian(fesolver,feop,sols[1:nsnaps_system],params[1:nsnaps_system],times;i)
basis_space = tpod(Jac_offline[1])
interp_idx_space = get_interpolation_idx(basis_space)
@assert full_idx[interp_idx_space] == red_idx
err_jac = maximum(abs.(Jac-Jac_full[interp_idx_space,:]))
println("Jacobian #$i difference for selected triangulation is $err_jac")

coeff = mdeim_solve!(scache[1],rbjact.mdeim_interpolation,Jac)
coeff_ok = basis_space'*Jac_full
err_coeff = maximum(abs.(coeff-coeff_ok))
