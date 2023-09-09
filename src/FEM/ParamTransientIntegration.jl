struct PIntegrand
  object::Function
  meas::Measure
end

const ∫ₚ = PIntegrand

function collect_inputs(x::NTuple{3,Any})
  u,p,t = x
  pt = reshape(Iterators.product(p,t) |> collect,:)
  map((a,b)->(a,b),u,pt)
end

function Arrays.evaluate(int::PIntegrand,inputs)
  fields = map(i->int.object(i...),inputs)
  meas = int.meas
  integrate(fields,meas)
end

function CellData.integrate(fields::AbstractArray,quad::CellQuadrature)
  N = length(fields)
  f = testitem(fields)
  trian_f = get_triangulation(f)
  trian_x = get_triangulation(quad)

  msg = """\n
    Your are trying to integrate a CellField using a CellQuadrature defined on incompatible
    triangulations. Verify that either the two objects are defined in the same triangulation
    or that the triangulaiton of the CellField is the background triangulation of the CellQuadrature.
    """
  @check is_change_possible(trian_f,trian_x) msg

  b = lazy_map(f->change_domain(f,quad.trian,quad.data_domain_style),fields)
  x = get_cell_points(quad)
  bx = lazy_map(c->evaluate(c,x),b)
  ptmap = PTMap(IntegrationMap())
  if quad.data_domain_style == PhysicalDomain() &&
            quad.integration_domain_style == PhysicalDomain()
    lazy_map(ptmap,bx,fill(quad.cell_weight,N))
  elseif quad.data_domain_style == ReferenceDomain() &&
            quad.integration_domain_style == PhysicalDomain()
    cell_map = get_cell_map(quad.trian)
    cell_Jt = lazy_map(∇,cell_map)
    cell_Jtx = lazy_map(evaluate,cell_Jt,quad.cell_point)
    lazy_map(ptmap,bx,fill(quad.cell_weight,N),fill(cell_Jtx,N))
  elseif quad.data_domain_style == ReferenceDomain() &&
            quad.integration_domain_style == ReferenceDomain()
    cell_map = Fill(GenericField(identity),length(bx))
    cell_Jt = lazy_map(∇,cell_map)
    cell_Jtx = lazy_map(evaluate,cell_Jt,quad.cell_point)
    lazy_map(ptmap,bx,fill(quad.cell_weight,N),fill(cell_Jtx,N))
  else
    @notimplemented
  end
end
