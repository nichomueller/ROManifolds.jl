obj = ∫(∇(dv[1])⊙∇(du[1])).object
quad = dΩ.quad
f = CellField(obj,quad.trian,quad.data_domain_style)
trian_f = get_triangulation(f)
trian_x = get_triangulation(quad)
b = change_domain(f,quad.trian,quad.data_domain_style)
x = get_cell_points(quad)
bx = b(x)
cell_map = get_cell_map(quad.trian)
cell_Jt = lazy_map(∇,cell_map)
cell_Jtx = lazy_map(evaluate,cell_Jt,quad.cell_point)
k = IntegrationMap()

function Gridap.return_value(
  k::IntegrationMap,
  fx::ArrayBlock{A,N} where A,
  i::Int,
  args...) where N

  if fx.touched[i]
    return_value(k,fx.array[i],args...)
  else
    fxi = testitem(fx)
    return_value(k,fxi,args...)
  end
end

# _check_blocks(args...) = true

# function _check_blocks(
#   array_a::ArrayBlock{Aa,N},
#   array_b::ArrayBlock{Ab,N}) where {Aa,Ab,N}

#   array_a.touched == array_b.touched
# end

# function (+)(a::DomainContribution,b::DomainContribution)
#   array_a = first(values(a.dict))
#   array_b = first(values(b.dict))
#   c = copy(a)
#   if _check_blocks(array_a,array_b)
#     for (trian,array) in b.dict
#       add_contribution!(c,trian,array)
#     end
#   end
#   c
# end

function my_integrate(f,b::Gridap.CellData.GenericMeasure)
  c = my_integrate(f,b.quad)
  cont = DomainContribution()
  add_contribution!(cont,b.quad.trian,c)
  cont
end

function my_integrate(f::CellField,quad::CellQuadrature)
  trian_f = get_triangulation(f)
  trian_x = get_triangulation(quad)

  msg = """\n
    Your are trying to integrate a CellField using a CellQuadrature defined on incompatible
    triangulations. Verify that either the two objects are defined in the same triangulation
    or that the triangulaiton of the CellField is the background triangulation of the CellQuadrature.
    """
  @check is_change_possible(trian_f,trian_x) msg

  b = my_change_domain(f,quad.trian,quad.data_domain_style)
  x = get_cell_points(quad)
  bx = b(x)
  if quad.data_domain_style == PhysicalDomain() &&
            quad.integration_domain_style == PhysicalDomain()
    lazy_map(IntegrationMap(),bx,quad.cell_weight)
  elseif quad.data_domain_style == ReferenceDomain() &&
            quad.integration_domain_style == PhysicalDomain()
    cell_map = get_cell_map(quad.trian)
    cell_Jt = lazy_map(∇,cell_map)
    cell_Jtx = lazy_map(evaluate,cell_Jt,quad.cell_point)
    lazy_map(IntegrationMap(),bx,quad.cell_weight,cell_Jtx)
  elseif quad.data_domain_style == ReferenceDomain() &&
            quad.integration_domain_style == ReferenceDomain()
    cell_map = Fill(GenericField(identity),length(bx))
    cell_Jt = lazy_map(∇,cell_map)
    cell_Jtx = lazy_map(evaluate,cell_Jt,quad.cell_point)
    lazy_map(IntegrationMap(),bx,quad.cell_weight,cell_Jtx)
  else
    @notimplemented
  end
end

my_change_domain(args...;kwargs...) = change_domain(args...;kwargs...)

function my_change_domain(f::Gridap.CellData.OperationCellField,target_trian::Triangulation,target_domain::DomainStyle)
  args = map(i->my_change_domain(i,target_trian,target_domain),f.args)
  Gridap.CellData.OperationCellField(f.op,args...)
end

function my_change_domain(
  a::Gridap.MultiField.MultiFieldFEBasisComponent,
  ttrian::Triangulation,
  tdomain::DomainStyle)

  change_domain(a.single_field,ttrian,tdomain)
end

fs(du,v) = my_integrate(∫(∇(v)⊙∇(du)).object,dΩ)
fm((du,dp),(v,q)) = my_integrate(∫(∇(v)⊙∇(du)).object,dΩ)

t = 1.
dvu = get_fe_basis(test_u)
duu = get_trial_fe_basis(trial_u(t))
dv = get_fe_basis(test)
du = get_trial_fe_basis(trial(t))

dcfs = fs(duu,dvu)
dcfm = fm(du,dv)
trian = get_triangulation(test)

Jfs = assemble_matrix(dcfs,trial_u(t),test_u)
Jfm = assemble_matrix(dcfm,trial_u(t),test_u)

dcfm.dict[trian]
