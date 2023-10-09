rbres,rbjac = rbrhs,rblhs
sols_test,params_test = load_test(info,feop,fesolver)
res_cache,jac_cache = allocate_online_cache(feop,fesolver,rbspace,sols_test,params_test)

# RESIDUALS
trians = get_domains(rbres)
trian,full_meas = Ω,dΩ #Γn,dΓn
rbrest = rbres[trian]
cache, = res_cache

rcache,scache... = cache

times = get_times(fesolver)
ndofs = num_free_dofs(feop.test)
setsize!(rcache,(ndofs,))

red_idx = rbrest.integration_domain.idx
red_times = rbrest.integration_domain.times
red_meas = rbrest.integration_domain.meas
red_trian = get_triangulation(red_meas)

dv = get_fe_basis(feop.test)
ode_op = get_algebraic_operator(feop)
ode_cache = allocate_cache(ode_op,params_test,times)
ode_cache = update_cache!(ode_cache,ode_op,params_test,times)
Xh, = ode_cache
dxh = ()
_xh = (sols_test,sols_test-sols_test)
for i in 2:get_order(feop)+1
  dxh = (dxh...,EvaluationFunction(Xh[i],_xh[i]))
end
xh = TransientCellField(EvaluationFunction(Xh[1],_xh[1]),dxh)
dc = feop.res(params_test,times,xh,dv)[full_meas]
vecdata = collect_cell_vector(feop.test,dc,trian)
dcred = feop.res(params_test,times,xh,dv)[red_meas]
vecdata_red = collect_cell_vector(feop.test,dcred,red_trian)

# stuff from AffineDecomposition
cell_dof_ids = get_cell_dof_ids(feop.test,trian)
red_integr_cells = find_cells(red_idx,cell_dof_ids)
red_trian = view(trian,red_integr_cells)
red_meas = get_measure(feop,red_trian)

_f(x,y) = maximum(abs.(x-y))
maximum(map(_f,vecdata[1][1][1][red_integr_cells],vecdata_red[1][1][1][:]))


# JACOBIANS
i = 2
trians = get_domains(rbjac[i])
trian,full_meas = Ω,dΩ
rbjact = rbjac[i][trian]
cache, = jac_cache

jcache,scache... = cache

times = get_times(fesolver)
ndofs_row = num_free_dofs(feop.test)
ndofs_col = num_free_dofs(get_trial(feop)(nothing,nothing))
setsize!(jcache,(ndofs_row,ndofs_col))

red_idx = rbjact.integration_domain.idx
red_times = rbjact.integration_domain.times
red_meas = rbjact.integration_domain.meas
red_trian = get_triangulation(red_meas)

dv = get_fe_basis(feop.test)
du = get_trial_fe_basis(get_trial(feop)(nothing,nothing))
ode_op = get_algebraic_operator(feop)
ode_cache = allocate_cache(ode_op,params_test,times)
ode_cache = update_cache!(ode_cache,ode_op,params_test,times)
Xh, = ode_cache
dxh = ()
_xh = (sols_test,sols_test-sols_test)
for i in 2:get_order(feop)+1
  dxh = (dxh...,EvaluationFunction(Xh[i],_xh[i]))
end
xh = TransientCellField(EvaluationFunction(Xh[1],_xh[1]),dxh)
dc = feop.jacs[i](params_test,times,xh,du,dv)[full_meas]
matdata = collect_cell_matrix(get_trial(feop)(params_test,times),feop.test,dc,trian)
dcred = feop.jacs[i](params_test,times,xh,du,dv)[red_meas]
matdata_red = collect_cell_matrix(get_trial(feop)(params_test,times),feop.test,dcred,red_trian)

# stuff from AffineDecomposition
cell_dof_ids = get_cell_dof_ids(feop.test,trian)
red_integr_cells = find_cells(red_idx,cell_dof_ids)
red_trian = view(trian,red_integr_cells)
red_meas = get_measure(feop,red_trian)

_f(x,y) = maximum(abs.(x-y))
if i == 1
  maximum(map(_f,matdata[1][1][1][red_integr_cells],matdata_red[1][1][1][:]))
else
  maximum(map(_f,matdata[1][1][red_integr_cells],matdata_red[1][1][:]))
end


# something not right when i=2
i = 2
combine_projections = (x,y) -> i == 1 ? θ*x+(1-θ)*y : θ*x-θ*y
jacs,trians = collect_jacobians_for_trian(fesolver,feop,sols[1:10],params,times;i)
nzm,trian = jacs[1],[trians...][1]
basis_space,basis_time = compress(nzm;ϵ=info.ϵ)

M = assemble_matrix((u,v)->∫(u*v)dΩ,trial(nothing,nothing),test)/(θ*dt)
# bs = basis_space.nonzero_val
# bt = basis_time
# nparams = 10
# m2 = change_mode(nzm.nonzero_val,nparams)
# @check bs*bs'*nzm.nonzero_val ≈ nzm.nonzero_val
# @check bt*bt'*m2 ≈ m2
# @check NnzVector(M).nonzero_val ≈ nzm.nonzero_val[:,1]
# @check abs.(bs) ≈ abs.(tpod(nzm.nonzero_val[:,1:1]))

proj_bs,proj_bt = project_space_time(basis_space,basis_time,rbspace,rbspace;combine_projections)
interp_idx_space = get_interpolation_idx(basis_space)
interp_idx_time = get_interpolation_idx(basis_time)
entire_interp_idx_space = recast_idx(nzm,interp_idx_space)
entire_interp_idx_rows,_ = vec_to_mat_idx(entire_interp_idx_space,nzm.nrows)

irow,icol = vec_to_mat_idx(nzm.nonzero_idx,nzm.nrows)
ncols = maximum(icol)
BS = sparse(irow,icol,bs[:],nzm.nrows,ncols)
COEFF_OK = basis_space'*nzm.nonzero_val[:,1]
@check proj_bs[1] ≈ rbspace.basis_space'*BS*rbspace.basis_space
@check proj_bt[1] ≈ bt
@check nzm.nonzero_val[:,1] ≈ bs*COEFF_OK

interp_bs = basis_space[interp_idx_space,:]
lu_interp = if info.st_mdeim
  interp_bt = basis_time[interp_idx_time,:]
  interp_bst = LinearAlgebra.kron(interp_bt,interp_bs)
  lu(interp_bst)
else
  lu(interp_bs)
end

cell_dof_ids = get_cell_dof_ids(feop.test,trian)
red_integr_cells = find_cells(entire_interp_idx_rows,cell_dof_ids)
red_trian = view(trian,red_integr_cells)
red_meas = get_measure(feop,red_trian)
red_times = st_mdeim ? times[interp_idx_time] : times
integr_domain = RBIntegrationDomain(red_meas,red_times,entire_interp_idx_space)

opposite_cells = setdiff(collect(eachindex(cell_dof_ids)),red_integr_cells)
for idx in entire_interp_idx_rows
  @check any([idx ∈ cell_dof_ids[cell] for cell in red_integr_cells])
  @check !any([idx ∈ cell_dof_ids[cell] for cell in opposite_cells])
end

nnzidx = findnz(M[:])[1]
Mred = assemble_matrix((u,v)->∫(u*v)red_meas,trial(nothing,nothing),test)/(θ*dt)
norm(Mred[nnzidx][interp_idx_space] - M[nnzidx][interp_idx_space])

cache, = jac_cache
trian,full_meas = Ω,dΩ
rbjact = rbjac[i][trian]
# red_jac2 = assemble_lhs!(cache[1],feop,fesolver,rbjact,sols_test,params_test;i=2)
# coeff = mdeim_solve!(cache[2],rbjact,red_jac2;st_mdeim)
times = get_times(fesolver)
ndofs_row = num_free_dofs(feop.test)
ndofs_col = num_free_dofs(get_trial(feop)(nothing,nothing))
setsize!(cache[1],(ndofs_row,ndofs_col))

red_idx = rbjact.integration_domain.idx
red_times = rbjact.integration_domain.times
red_meas = rbjact.integration_domain.meas

ode_op = get_algebraic_operator(feop)
ode_cache = allocate_cache(ode_op,params_test,red_times)

if length(red_times) < length(times)
  A = get_array(cache;len=length(red_times)*length(params_test))
  time_idx = findall(x->x in red_times,times)
  _sols = map(x->getindex(x,time_idx),sols)
else
  A = get_array(cache[1])
  _sols = sols_test
end
jac_i = collect_jacobians!(A,fesolver,ode_op,_sols,params_test,red_times,ode_cache,red_idx,red_meas;i)
coeff = mdeim_solve!(cache[2:3],rbjact,jac_i;st_mdeim)

coeff_param1 = coeff[1]
coeff_param1_time1 = coeff_param1[1]
jac_i_mdeim_red_param1_time1 = basis_space*coeff_param1_time1

irow,icol = vec_to_mat_idx(nzm.nonzero_idx,nzm.nrows)
ncols = maximum(icol)
jac_i_mdeim_param1_time1 = sparse(irow,icol,jac_i_mdeim_red_param1_time1[:],nzm.nrows,ncols)
jac_i_param1_time1 = sparse(irow,icol,nzm.nonzero_val[:,1],nzm.nrows,ncols)
maximum(abs.(jac_i_mdeim_param1_time1 - jac_i_param1_time1))

M = assemble_matrix((u,v)->∫(u*v)dΩ,trial(nothing,nothing),test)
isapprox(M/(dt*θ),jac_i_param1_time1)

#
t = get_domains(rbrhs1)
trians1 = [t...]

ad1 = rbrhs[Ω]
ad2 = rbrhs1[trians1[1]]

trians1[1] == Ω
trians1[2] == Γn
ad1.basis_space == ad2.basis_space
ad1.basis_time == ad2.basis_time
ad1.integration_domain.idx == ad2.integration_domain.idx
ad1.integration_domain.times == ad2.integration_domain.times
get_triangulation(ad1.integration_domain.meas) == get_triangulation(ad2.integration_domain.meas)
ad1.mdeim_interpolation.L == ad2.mdeim_interpolation.L
ad1.mdeim_interpolation.U == ad2.mdeim_interpolation.U

for i = 1:2
  t = get_domains(rblhs1[1])
  trians1 = [t...]

  ad1 = rblhs[i][Ω]
  ad2 = rblhs1[i][trians1[1]]

  trians1[1] == Ω # false
  ad1.basis_space == ad2.basis_space
  ad1.basis_time == ad2.basis_time
  ad1.integration_domain.idx == ad2.integration_domain.idx
  ad1.integration_domain.times == ad2.integration_domain.times
  get_triangulation(ad1.integration_domain.meas) == get_triangulation(ad2.integration_domain.meas)
  ad1.mdeim_interpolation.L == ad2.mdeim_interpolation.L
  ad1.mdeim_interpolation.U == ad2.mdeim_interpolation.U
end
