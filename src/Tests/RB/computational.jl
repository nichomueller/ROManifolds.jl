begin
  root = pwd()
  mesh = "cube2x2.json"
  bnd_info = Dict("dirichlet" => [1,2,3,4,5,7,8],"neumann" => [6])
  test_path = "$root/tests/poisson/unsteady/$mesh"
  order = 1
  degree = 2

  model = get_discrete_model(test_path,mesh,bnd_info)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  Γn = BoundaryTriangulation(model,tags=["neumann"])
  dΓn = Measure(Γn,degree)

  ranges = fill([1.,10.],3)
  sampling = UniformSampling()
  pspace = PSpace(ranges,sampling)

  a(x,μ,t) = exp((sin(t)+cos(t))*x[1]/sum(μ))
  a(μ,t) = x->a(x,μ,t)
  aμt(μ,t) = PTFunction(a,μ,t)

  f(x,μ,t) = 1.
  f(μ,t) = x->f(x,μ,t)
  fμt(μ,t) = PTFunction(f,μ,t)

  h(x,μ,t) = abs(cos(t/μ[3]))
  h(μ,t) = x->h(x,μ,t)
  hμt(μ,t) = PTFunction(h,μ,t)

  g(x,μ,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
  g(μ,t) = x->g(x,μ,t)

  u0(x,μ) = 0
  u0(μ) = x->u0(x,μ)
  u0μ(μ) = PFunction(u0,μ)

  res(μ,t,u,v) = ∫ₚ(v*∂ₚt(u),dΩ) + ∫ₚ(aμt(μ,t)*∇(v)⊙∇(u),dΩ) - ∫ₚ(fμt(μ,t)*v,dΩ) - ∫ₚ(hμt(μ,t)*v,dΓn)
  jac(μ,t,u,du,v) = ∫ₚ(aμt(μ,t)*∇(v)⋅∇(du),dΩ)
  jac_t(μ,t,u,dut,v) = ∫ₚ(v*dut,dΩ)

  reffe = ReferenceFE(lagrangian,Float,order)
  test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
  trial = PTTrialFESpace(test,g)
  feop = AffinePTFEOperator(res,jac,jac_t,pspace,trial,test)
  t0,tf,dt,θ = 0.,0.3,0.005,0.5
  uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
  fesolver = PThetaMethod(LUSolver(),uh0μ,θ,dt,t0,tf)

  ϵ = 1e-4
  load_solutions = false
  save_solutions = true
  load_structures = false
  save_structures = true
  norm_style = :l2
  nsnaps_state = 50
  nsnaps_mdeim = 20
  nsnaps_test = 10
  st_mdeim = false
  postprocess = true
  info = RBInfo(test_path;ϵ,norm_style,nsnaps_state,nsnaps_mdeim,nsnaps_test,st_mdeim,postprocess)
end

sols,params = load(info,(Snapshots,Table))
rbspace = load(info,RBSpace)
rbrhs,rblhs = load(info,(RBVecAlgebraicContribution{Float},Vector{RBMatAlgebraicContribution{Float}}))

rbrest = rbrhs[[get_domains(rbrhs)...][1]]
red_meas = rbrest.integration_domain.meas

μ = rand(3)
t = 0.1
dv = get_fe_basis(test)
du = get_trial_fe_basis(trial(nothing,nothing))
int = ∫(a(μ,t)*∇(dv)⊙∇(du))
ξ = int.object

@btime int*dΩ
quad = dΩ.quad
red_quad = red_meas.quad
x = get_cell_points(quad)

@btime int*red_meas

b = change_domain(ξ,quad.trian,quad.data_domain_style)
red_b = change_domain(ξ,red_quad.trian,red_quad.data_domain_style)

@time bx = b(x)
@time red_bx = red_b(x)

#########
# b(x)
# evaluate(b,x)
c = return_cache(b,x)
# evaluate!(c,b,x)
ax = map(i->i(x),b.args)
# b.args[1](x)
# b.args[1].args[1](x)
# _f, _x = _to_common_domain(b.args[1].args[1],x)
trian_x = get_triangulation(x)
# f_on_trian_x = change_domain(b.args[1].args[1],trian_x,DomainStyle(x))
_a = b.args[1].args[1]
_b = red_b.args[1].args[1]
# change_domain(_a,get_triangulation(_a),DomainStyle(_a),trian_x,DomainStyle(x))
@time begin
  D = num_cell_dims(get_triangulation(_a))
  sglue = get_glue(get_triangulation(_a),Val(D))
  tglue = get_glue(trian_x,Val(D))
end
@time begin
  _D = num_cell_dims(get_triangulation(_b))
  _sglue = get_glue(get_triangulation(_b),Val(_D))
  _tglue = get_glue(trian_x,Val(D))
end
CellData.change_domain_ref_ref(_a,trian_x,sglue,tglue)
# cell_field = get_data(_f)
# cell_point = get_data(_x)
# lazy_map(evaluate,cell_field,cell_point)
# lazy_map(Fields.BroadcastingFieldOpMap(b.op.op),ax...)
#########
#########
# red_b(x)
# evaluate(red_b,x)
c = return_cache(red_b,x)
# evaluate!(c,red_bx,x)
ax = map(i->i(x),red_b.args)
# lazy_map(Fields.BroadcastingFieldOpMap(red_b.op.op),ax...)
#########




@time cell_map = get_cell_map(quad.trian)
@time red_cell_map = get_cell_map(red_quad.trian)

@time cell_Jt = lazy_map(∇,cell_map)
@time red_cell_Jt = lazy_map(∇,red_cell_map)

@time cell_Jtx = lazy_map(evaluate,cell_Jt,quad.cell_point)
@time red_cell_Jtx = lazy_map(evaluate,red_cell_Jt,red_quad.cell_point)

@time lazy_map(IntegrationMap(),bx,quad.cell_weight,cell_Jtx)
@time lazy_map(IntegrationMap(),red_bx,red_quad.cell_weight,red_cell_Jtx)

aa = int*dΩ
bb = int*red_meas

@time cm1 = collect_cell_matrix(trial(nothing,nothing),test,aa)
@time cm2 = collect_cell_matrix(trial(nothing,nothing),test,bb)

@time assemble_matrix(feop.assem,cm1)
@time assemble_matrix(feop.assem,cm2)


idx = rand(1:1360,10)
t1 = Triangulation(model,idx)
t2 = view(Ω,idx)

m1 = Measure(t1,2)
m2 = Measure(t2,2)

@time int*dΩ
@time int*m1
@time int*m2

propertynames(Ω)

propertynames(m2)

t3 = RBTriangulation(Ω,idx)
m3 = Measure(t3,2)
@time int*m3
@time int*dΩ

quad = dΩ.quad
red_quad = m3.quad

b = change_domain(ξ,quad.trian,quad.data_domain_style)
red_b = change_domain(ξ,red_quad.trian,red_quad.data_domain_style)

@time bx = b(x)
@time red_bx = red_b(x)


# analize change_domain(ξ,quad.trian,quad.data_domain_style)
change_domain(ξ.args[1],quad.trian,quad.data_domain_style)
change_domain(ξ.args[1].args[1],quad.trian,quad.data_domain_style)
η = ξ.args[1].args[1]
change_domain(η,get_triangulation(η),DomainStyle(η),quad.trian,quad.data_domain_style)

D = num_cell_dims(get_triangulation(η))
sglue = get_glue(get_triangulation(η),Val(D))
tglue = get_glue(quad.trian,Val(D))
CellData.change_domain_ref_ref(η,quad.trian,sglue,tglue)

tglue = get_glue(red_quad.trian,Val(D))
change_domain_ref_ref(η,red_quad.trian,sglue,tglue)

trian = red_quad.trian
tface_to_mface = trian.tface_to_mface
tface_to_mface_map = Fill(GenericField(identity),num_cells(trian))
if isa(tface_to_mface,IdentityVector) && num_faces(trian.model,Dt) == num_cells(trian)
  mface_to_tface = tface_to_mface
else
  nmfaces = num_faces(trian.model,D)
  mface_to_tface = PosNegPartition(tface_to_mface,Int32(nmfaces))
end

grid_topo = get_grid_topology(trian.model)


t2 = view(Ω,idx)
active_model = Geometry.compute_active_model(t2)
ΩΩ = Triangulation(active_model)
dΩΩ = Measure(ΩΩ,2)
red_test = TestFESpace(active_model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
red_trial = PTTrialFESpace(red_test,g)
red_dv = get_fe_basis(red_test)
red_du = get_trial_fe_basis(red_trial(nothing,nothing))
red_int = ∫(a(μ,t)*∇(red_dv)⊙∇(red_du))
@time red_int*dΩΩ
red_quad = dΩΩ.quad
red_x = get_cell_points(red_quad)
(a(μ,t)*∇(red_dv)⊙∇(red_du))(red_x)
(a(μ,t)*∇(dv)⊙∇(du))(x)

idx = Int32.(idx)
_model = RBDiscreteModelPortion(model,idx)
red_test = TestFESpace(_model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
red_trial = PTTrialFESpace(red_test,g)
tt = Triangulation(_model)
mm = Measure(tt,2)
qq = mm.quad
red_x = get_cell_points(qq)
red_dv = get_fe_basis(red_test)
red_du = get_trial_fe_basis(red_trial(nothing,nothing))
red_int = ∫(a(μ,t)*∇(red_dv)⊙∇(red_du))
@time red_int*mm

@time int*dΩ

arr = (red_int*mm)[tt]
arr_ok = (int*dΩ)[Ω]
for (i,iidx) = enumerate(idx)
  arr_ok[iidx] ≈ arr[i]
end

@time (a(μ,t)*∇(red_dv)⊙∇(red_du))(red_x)
@time (a(μ,t)*∇(dv)⊙∇(du))(x)

@time cell_∇a = lazy_map(Broadcasting(∇),get_data(dv))
@time cell_map = get_cell_map(get_triangulation(dv))
# @time gg = lazy_map(Broadcasting(push_∇),cell_∇a,cell_map)
@time cell_Jt = lazy_map(∇,cell_map)
@time cell_invJt = lazy_map(Operation(pinvJt),cell_Jt)
@time lazy_map(Broadcasting(Operation(⋅)),cell_invJt,cell_∇a)

@time _cell_∇a = lazy_map(Broadcasting(∇),get_data(red_dv))
@time _cell_map = get_cell_map(get_triangulation(red_dv))
# @time _gg = lazy_map(Broadcasting(push_∇),_cell_∇a,_cell_map)
@time _cell_Jt = lazy_map(∇,_cell_map)
@time _cell_invJt = lazy_map(Operation(pinvJt),_cell_Jt)
@time lazy_map(Broadcasting(Operation(⋅)),_cell_invJt,_cell_∇a)

###############
@which red_int*mm
red_quad = mm.quad
@time red_b = change_domain(red_int.object,red_quad.trian,red_quad.data_domain_style)
red_x = get_cell_points(red_quad)
@time red_bx = red_b(red_x)
red_cell_map = get_cell_map(red_quad.trian)
@time red_cell_Jt = lazy_map(∇,red_cell_map)
@time red_cell_Jtx = lazy_map(evaluate,red_cell_Jt,red_quad.cell_point)
@time lazy_map(IntegrationMap(),red_bx,red_quad.cell_weight,red_cell_Jtx)

quad = dΩ.quad
@time b = change_domain(int.object,quad.trian,quad.data_domain_style)
x = get_cell_points(quad)
@time bx = b(x)
cell_map = get_cell_map(quad.trian)
@time cell_Jt = lazy_map(∇,cell_map)
@time cell_Jtx = lazy_map(evaluate,cell_Jt,quad.cell_point)
@time lazy_map(IntegrationMap(),bx,quad.cell_weight,cell_Jtx)

@time c = integrate(int.object,dΩ.quad)

@time begin
  trian_f = get_triangulation(int.object)
  trian_x = get_triangulation(quad)

  @check is_change_possible(trian_f,trian_x) msg
end

@time begin
  b = change_domain(int.object,quad.trian,quad.data_domain_style)
  x = get_cell_points(quad)
  bx = b(x)
  cell_map = get_cell_map(quad.trian)
  cell_Jt = lazy_map(∇,cell_map)
  cell_Jtx = lazy_map(evaluate,cell_Jt,quad.cell_point)
  lazy_map(IntegrationMap(),bx,quad.cell_weight,cell_Jtx)
end

################################################################################
idx = Int32.(idx)
_model = RBDiscreteModelPortion(model,idx)
red_test = TestFESpace(_model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
red_trial = PTTrialFESpace(red_test,g)
tt = Triangulation(_model)
mm = Measure(tt,2)
qq = mm.quad
red_x = get_cell_points(qq)
red_dv = get_fe_basis(red_test)
red_du = get_trial_fe_basis(red_trial(nothing,nothing))
red_int = ∫(a(μ,t)*∇(red_dv)⊙∇(red_du))
@time red_int*mm

@time cell_∇a = lazy_map(Broadcasting(∇),get_data(dv))
@time cell_map = get_cell_map(get_triangulation(dv))
# @time gg = lazy_map(Broadcasting(push_∇),cell_∇a,cell_map)
@time cell_Jt = lazy_map(∇,cell_map)
gradients = cell_map.args[1]
lazy_map(constant_field,gradients)
# @time cell_invJt = lazy_map(Operation(pinvJt),cell_Jt)
# @time lazy_map(Broadcasting(Operation(⋅)),cell_invJt,cell_∇a)

@time _cell_∇a = lazy_map(Broadcasting(∇),get_data(red_dv))
@time _cell_map = get_cell_map(get_triangulation(red_dv))
# @time _gg = lazy_map(Broadcasting(push_∇),_cell_∇a,_cell_map)
@time _cell_Jt = lazy_map(∇,_cell_map)
_gradients = _cell_map.args[1]
lazy_map(constant_field,_gradients)
# @time _cell_invJt = lazy_map(Operation(pinvJt),_cell_Jt)
# @time lazy_map(Broadcasting(Operation(⋅)),_cell_invJt,_cell_∇a)

# lazy_map(Reindex(get_cell_map(grid)),idx)
k = Reindex(get_cell_map(grid))
fi = map(testitem,(idx,))
T = return_type(k,fi...)
# lazy_map(k,T,idx)
i_to_maps = k.values.maps
i_to_args = k.values.args
j_to_maps = lazy_map(Reindex(i_to_maps),eltype(i_to_maps),idx)
j_to_args = map(i_to_fk->lazy_map(Reindex(i_to_fk),eltype(i_to_fk),idx), i_to_args)
LazyArray(T,j_to_maps,j_to_args...)

######
μ = rand(3)
t = 0.1
dv = get_fe_basis(test)
du = get_trial_fe_basis(trial(nothing,nothing))
int = ∫(a(μ,t)*∇(dv)⊙∇(du))
idx = rand(1:1360,10)
_model = DiscreteModelPortion(model,idx)
_Ω = Triangulation(_model)
d_Ω = Measure(_Ω,degree)
_test = TestFESpace(_model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
_trial = PTTrialFESpace(_test,g)
_dv = get_fe_basis(_test)
_du = get_trial_fe_basis(_trial(nothing,nothing))
_int = ∫(a(μ,t)*∇(_dv)⊙∇(_du))
@time int*dΩ
@time _int*d_Ω

Profile.clear()

@profile begin
  for _ in 1:1000
    ∇(dv)# int*dΩ
  end
end
open("/tmp/prof.txt", "w") do s
  Profile.print(IOContext(s, :displaysize => (24, 500)))
end

Profile.clear()

@profile begin
  for _ in 1:1000
    ∇(_dv) #_int*d_Ω
  end
end
open("/tmp/_prof.txt", "w") do s
  Profile.print(IOContext(s, :displaysize => (24, 500)))
end

############
μ = rand(3)
t = 0.1
dv = get_fe_basis(test)
du = get_trial_fe_basis(trial(nothing,nothing))
int = ∫(a(μ,t)*∇(dv)⊙∇(du))
idx = rand(1:1360,10)
_model = DiscreteModelPortion(model,idx)
_grid = get_grid(_model)
_Ω = Triangulation(_model)
_dΩ = Measure(_Ω,degree)
_test = TestFESpace(_model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
_trial = PTTrialFESpace(_test,g)
_dv = get_fe_basis(_test)
_du = get_trial_fe_basis(_trial(nothing,nothing))
_int = ∫(a(μ,t)*∇(_dv)⊙∇(_du))

grid = get_grid(model)
node_coordinates = collect1d(get_node_coordinates(grid))
cell_node_ids = get_cell_node_ids(grid)[idx]
reffes = get_reffes(grid)
cell_types = get_cell_type(grid)[idx]
# cell_map = lazy_map(Reindex(get_cell_map(grid)),ids)
# cell_map = get_cell_map(grid)[idx]
cell_map = Geometry._compute_cell_map(node_coordinates,cell_node_ids,reffes,cell_types,Geometry.get_has_affine_map(reffes))
orien = OrientationStyle(grid)
__grid = RBGridPortion(Int32.(idx),node_coordinates,cell_node_ids,reffes,cell_types,cell_map,orien)
__model = RBDiscreteModelPortion(model,__grid)
__Ω = Triangulation(__model)
__dΩ = Measure(__Ω,degree)
__test = TestFESpace(__model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
__trial = PTTrialFESpace(__test,g)
__dv = get_fe_basis(__test)
__du = get_trial_fe_basis(__trial(nothing,nothing))
__int = ∫(a(μ,t)*∇(__dv)⊙∇(__du))

@time ∇(dv)
@time ∇(_dv)
@time ∇(__dv)

@time ∫(a(μ,t)*∇(dv)⊙∇(du))
@time ∫(a(μ,t)*∇(_dv)⊙∇(_du))
@time ∫(a(μ,t)*∇(__dv)⊙∇(__du))

@time ∫(a(μ,t)*∇(dv)⊙∇(du))dΩ
@time ∫(a(μ,t)*∇(_dv)⊙∇(_du))_dΩ
@time ∫(a(μ,t)*∇(__dv)⊙∇(__du))__dΩ

aa = a(μ,t)*∇(dv)
bb = ∇(du)
@which (a(μ,t)*∇(dv)⊙∇(du))
@which Operation(inner)(aa,bb)
@which evaluate(Operation(inner),aa,bb)
c = return_cache(Operation(inner),aa,bb)
@which evaluate!(c,Operation(inner),aa,bb)
@which _operate_cellfields(Operation(inner),aa,bb)
b = CellData._to_common_domain(aa,bb)

# _to_common_domain analysis

target_domain = ReferenceDomain()
trian_candidates = unique(objectid,map(get_triangulation,(aa,bb)))
target_trian = first(trian_candidates)
@which change_domain(aa,target_trian,target_domain)
# @which change_domain(bb,target_trian,target_domain)
args = map(i->change_domain(i,target_trian,target_domain),aa.args)
@time OperationCellField(aa.op,args...)

@assert length(args) > 0
trian = get_triangulation(first(args))
domain_style = DomainStyle(first(args))
@check all( map(i->DomainStyle(i)==domain_style,args) )
@time x = _get_cell_points(args...)
@time ax = map(i->i(x),args)
@which args[2](x)
@which evaluate(args[2],x)
c = return_cache(args[2],x)
y = evaluate!(c,args[2],x)
@time _f, _x = _to_common_domain(args[2],x)
cell_field = get_data(_f)
cell_point = get_data(_x)
@which lazy_map(evaluate,cell_field,cell_point)
# @time fx = map(fi->lazy_map(evaluate,fi,cell_point),cell_field.args)
@which lazy_map(evaluate,cell_field.args[2],cell_point)
@assert Arrays._have_same_ptrs((cell_field.args[2],cell_point))
@time Arrays._lazy_map_compressed(cell_field.args[2],cell_point)
gg = (cell_field.args[2],cell_point)
@time vals = map(evaluate,map(gi->gi.values,gg)...)
@time evaluate(map(gi->gi.values,gg)[1])
@time evaluate(cell_field.args[2].values)

__aa = a(μ,t)*∇(__dv)
__bb = ∇(__du)
__target_domain = ReferenceDomain()
__trian_candidates = unique(objectid,map(get_triangulation,(__aa,__bb)))
__target_trian = first(__trian_candidates)
__args = map(i->change_domain(i,__target_trian,__target_domain),__aa.args)
@check all( map(i->DomainStyle(i)==domain_style,__args) )
@time __x = _get_cell_points(__args...)
@time __ax = map(i->i(__x),__args)
@which __args[2](__x)
@which evaluate(__args[2],__x)
__c = return_cache(__args[2],__x)
__y = evaluate!(__c,__args[2],__x)
@time ___f, ___x = _to_common_domain(__args[2],__x)
__cell_field = get_data(___f)
__cell_point = get_data(___x)
@which lazy_map(evaluate,__cell_field,__cell_point)
# @time __fx = map(fi->lazy_map(evaluate,fi,__cell_point),__cell_field.args)
__gg = (__cell_field.args[2],__cell_point)
@time __vals = map(evaluate,map(gi->gi.values,__gg)...)


###############
###############
###############
μ = [rand(3) for _ = 1:10]
t = rand(10)
@time ∫(aμt(μ,t)*∇(dv)⋅∇(du))dΩ
@time ∫(aμt(μ,t)*∇(_dv)⋅∇(_du))_dΩ
@time ∫(aμt(μ,t)*∇(__dv)⋅∇(__du))__dΩ
@time ∫(a(μ[1],t[1])*∇(dv)⋅∇(du))dΩ

rtrian = view(Ω,idx)
rmeas = Measure(rtrian,2)
@time ∫(aμt(μ,t)*∇(dv)⋅∇(du))rmeas

###################
###################
###################
idx = Int32.(rand(1:1360,10))
red_grid = RBGridPortion(model,idx)
red_model = RBDiscreteModelPortion(model,red_grid)
grid = get_grid(red_model)
trian = Triangulation(red_model)


# aa = map(i->test.cell_dofs_ids[i],test.dirichlet_cells)
# ddofs = Vector{Int32}(undef,test.ndirichlet)
# for i in 1:test.ndirichlet
#   ddofs[i] =
# end
# uddofs = unique(ddofs)

# function dirichlet_dof_to_cell_ids(test)
#   dcells = map(i->test.cell_dofs_ids[i],test.dirichlet_cells)
#   map(cell -> count(x->x<0,cell),dcells)
# end

# ptrs = dirichlet_dof_to_cell_ids(test)
# ptrs_idx = ptrs[idx]

function get_reduced_dirichlet_ids(test,reduced_dirichlet_cells)
  dirichlet_cell_dofs_ids = map(i->test.cell_dofs_ids[i],reduced_dirichlet_cells)
  dirichlet_ids = vcat(filter.(x->x<0,dirichlet_cell_dofs_ids)...)
  @. dirichlet_ids *= -one(Int32)
  sorted_dirichlet_ids = sort(dirichlet_ids)
  unique(sorted_dirichlet_ids)
end

function get_reduced_free_ids(reduced_cell_dofs_ids)
  ids = vcat(filter.(x->x>0,reduced_cell_dofs_ids)...)
  sorted_ids = sort(ids)
  unique(sorted_ids)
end


#################
cell_to_parent_cell = idx
cell_dofs_ids = test.cell_dofs_ids[cell_to_parent_cell]
cell_is_dirichlet = test.cell_is_dirichlet[cell_to_parent_cell]
dirichlet_cells = intersect(test.dirichlet_cells,cell_to_parent_cell)
cell_basis = test.fe_basis.cell_basis[cell_to_parent_cell]
trian = Triangulation(model)
fe_basis = FESpaces.SingleFieldFEBasis(cell_basis,trian,FESpaces.TestBasis(),test.fe_basis.domain_style)
cell_dof_basis = test.fe_dof_basis.cell_dof[cell_to_parent_cell]
fe_dof_basis = CellDof(cell_dof_basis,trian,test.fe_dof_basis.domain_style)
ndirichlet = length(cell_is_dirichlet)
nfree = length(cell_to_parent_cell)
ntags = test.ntags
vector_type = test.vector_type
dirichlet_dofs = get_reduced_dirichlet_ids(test,dirichlet_cells)
free_dofs = get_reduced_free_ids(cell_dofs_ids)
dirichlet_dof_tag = test.dirichlet_dof_tag[dirichlet_dofs]
glue = test.metadata
if isnothing(glue)
  UnconstrainedFESpace(vector_type,nfree,ndirichlet,cell_dofs_ids,fe_basis,
    fe_dof_basis,cell_is_dirichlet,dirichlet_dof_tag,dirichlet_cells,ntags)
else
  dirichlet_dof_to_comp = glue.dirichlet_dof_to_comp[dirichlet_dofs]
  dirichlet_dof_to_node = glue.dirichlet_dof_to_node[dirichlet_dofs]
  free_dof_to_comp = glue.free_dof_to_comp[free_dofs]
  free_dof_to_node = glue.free_dof_to_node[free_dofs]
  node_and_comp_to_dof = glue.node_and_comp_to_dof[dofs]
  metadata = NodeToDofGlue(free_dof_to_node,free_dof_to_comp,diri_dof_to_node,
    diri_dof_to_comp,node_and_comp_to_dof)
  UnconstrainedFESpace(vector_type,nfree,ndirichlet,cell_dofs_ids,fe_basis,
    fe_dof_basis,cell_is_dirichlet,dirichlet_dof_tag,dirichlet_cells,ntags,glue)
end
