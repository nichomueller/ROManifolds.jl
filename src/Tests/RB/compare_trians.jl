dv = get_fe_basis(test)
du = get_trial_fe_basis(trial(nothing,nothing))
idx = collect(1:5:50)
μ = rand(3)
t = dt
params = realization(feop,10)
times = get_times(fesolver)

############################## BODYFITTED ###################################

Ξ = ReducedTriangulation(Ω,idx)
dΞ = Measure(Ξ,2)
ff = a(μ,t)*∇(dv)⋅∇(du)
trian_f = get_triangulation(ff)

quad = dΩ.quad
@time b = change_domain(ff,quad.trian,quad.data_domain_style)
@time cell_map = get_cell_map(quad.trian)
@time cell_Jt = lazy_map(∇,cell_map)
@time cell_Jtx = lazy_map(evaluate,cell_Jt,quad.cell_point)
x = get_cell_points(quad)
@time bx = b(x)
@time integral = lazy_map(IntegrationMap(),bx,quad.cell_weight,cell_Jtx)

_quad = dΞ.quad
@time _b = change_domain(ff,_quad.trian,_quad.data_domain_style)
@time _cell_map = get_cell_map(_quad.trian)
@time _cell_Jt = lazy_map(∇,_cell_map)
@time _cell_Jtx = lazy_map(evaluate,_cell_Jt,_quad.cell_point)
_x = get_cell_points(_quad)
@time _bx = _b(_x)
@time _integral = lazy_map(IntegrationMap(),_bx,_quad.cell_weight,_cell_Jtx)

@time ∫(a(μ,t)*∇(dv)⋅∇(du))dΩ
@time ∫(a(μ,t)*∇(dv)⋅∇(du))dΞ

@time ∫(aμt(params,times)*∇(dv)⋅∇(du))dΩ
@time ∫(aμt(params,times)*∇(dv)⋅∇(du))dΞ

Ωhat = view(Ω,idx)
dΩhat = Measure(Ωhat,2)
@time ∫(aμt(params,times)*∇(dv)⋅∇(du))dΩhat

_m = DiscreteModelPortion(model,idx)
_t = Triangulation(_m)
_dt = Measure(_t,2)
_test = TestFESpace(_m,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
_trial = PTTrialFESpace(_test,g)
_dv = get_fe_basis(_test)
_du = get_trial_fe_basis(_trial(nothing,nothing))
@time ∫(aμt(params,times)*∇(_dv)⋅∇(_du))_dt

_Ωhat = ReducedTriangulation(Ω,idx)
_dΩhat = Measure(Ωhat,2)
@time ∫(aμt(params,times)*∇(dv)⋅∇(du))_dΩhat

############################## BOUNDARY ###################################
@time dc = ∫(hμt(params,times)*dv)dΓn

Γnhat = view(Γn,idx)
dΓnhat = Measure(Γnhat,2)
@time dchat = ∫(hμt(params,times)*dv)dΓnhat

_Γnhat = ReducedTriangulation(Γn,idx)
_dΓnhat = Measure(_Γnhat,2)
@time _dchat = ∫(hμt(params,times)*dv)_dΓnhat


############################### NEW TESTS ######################################
ff = a(μ,t)*∇(dv)⋅∇(du)

quad = dΩ.quad
baseline_stats = @timed begin
  b = change_domain(ff,quad.trian,quad.data_domain_style)
  cell_map = get_cell_map(quad.trian)
  cell_Jt = lazy_map(∇,cell_map)
  cell_Jtx = lazy_map(evaluate,cell_Jt,quad.cell_point)
  x = get_cell_points(quad)
  bx = b(x)
  integral = lazy_map(IntegrationMap(),bx,quad.cell_weight,cell_Jtx)
end

reduced_stats = @timed begin
  b = change_domain(ff,quad.trian,quad.data_domain_style)
  cell_map = get_cell_map(quad.trian)
  cell_Jt = lazy_map(∇,cell_map)
  cell_Jtx = lazy_map(evaluate,cell_Jt,quad.cell_point)
  x = get_cell_points(quad)
  bx = b(x)
  _bx = lazy_map(Reindex(bx),idx)
  _w = lazy_map(Reindex(quad.cell_weight),idx)
  _Jx = lazy_map(Reindex(cell_Jtx),idx)
  _integral = lazy_map(IntegrationMap(),_bx,_w,_Jx)
end

Ξ = ReducedTriangulation(Ω,idx)
dΞ = Measure(Ξ,2)
_quad = dΞ.quad
prev_red_stats = @timed begin
  _b = change_domain(ff,_quad.trian,_quad.data_domain_style)
  _cell_map = get_cell_map(_quad.trian)
  _cell_Jt = lazy_map(∇,_cell_map)
  _cell_Jtx = lazy_map(evaluate,_cell_Jt,_quad.cell_point)
  _x = get_cell_points(_quad)
  __bx = _b(_x)
  __integral = lazy_map(IntegrationMap(),__bx,_quad.cell_weight,_cell_Jtx)
end

snaps_test,params_test = sols[end-nsnaps_test+1:end],params[end-nsnaps_test+1:end]
op = get_ptoperator(fesolver,feop,snaps_test,params_test)

form(μ,t,du,v) = ∫ₚ(aμt(μ,t)*∇(v)⋅∇(du),dΩ)
opf = form(params_test,times,du,dv)
rdΩ = ReducedMeasure(dΩ,idx)
Ωhat = view(Ω,idx)
dΩhat = Measure(Ωhat,2)
rdc = integrate(opf,rdΩ)#integrate(ff,rdΩ.meas.quad,rdΩ.cell_to_parent_cell)
vdc = integrate(opf.object,dΩhat)
dc = integrate(opf.object,dΩ)

ptrian = get_parent_triangulation(rdΩ)
trian = get_triangulation(rdΩ)

A = allocate_jacobian(op,op.vθ)#[1]
_A = copy(A)
@assert rdc[trian][idx] == dc[Ω][idx]
matdata = collect_cell_matrix(trial(nothing,nothing),test,rdc)
assemble_matrix_add!(A,feop.assem,matdata)

cell_mat,ttrian = move_contributions(rdc[trian],trian)
cell_mat_c = attach_constraints_cols(trial(nothing,nothing),cell_mat,ttrian)
cell_mat_rc = attach_constraints_rows(test,cell_mat_c,ttrian)

_matdata = collect_cell_matrix(trial(nothing,nothing),test,vdc)
assemble_matrix_add!(_A,feop.assem,_matdata)

_cell_mat,_ttrian = move_contributions(vdc[Ωhat],Ωhat)
_cell_mat_c = attach_constraints_cols(trial(nothing,nothing),_cell_mat,_ttrian)
_cell_mat_rc = attach_constraints_rows(test,_cell_mat_c,_ttrian)
