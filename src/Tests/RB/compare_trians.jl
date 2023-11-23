dv = get_fe_basis(test)
du = get_trial_fe_basis(trial(nothing,nothing))
idx = [1,10,20]
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
