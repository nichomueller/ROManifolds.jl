struct PIntegrand
  object::Function
  meas::Measure
end

const ∫ₚ = PIntegrand

function collect_inputs(p,t)
  reshape(Iterators.product(p,t) |> collect,:)
end

function collect_inputs(p::AbstractArray,t::Real)
  reshape(Iterators.product([p],t) |> collect,:)
end

function collect_inputs(u,p,t)
  pt = collect_inputs(p,t)
  try
    map((a,b)->(a,b...),u,pt)
  catch
    map((a,b)->(a,b...),(u,),pt)
  end
end

function Arrays.evaluate(int::PIntegrand,inputs;allocate=false)
  if allocate
    fields = int.object(first(inputs)...)
  else
    fields = map(i->int.object(i...),inputs)
  end
  meas = int.meas
  integrate(fields,meas)
end

struct PIntegrands <: GridapType
  dict::IdDict{Measure,PIntegrand}
end

PIntegrands() = PIntegrands(IdDict{Measure,PIntegrand}())

Base.copy(a::PIntegrands) = PIntegrands(copy(a.dict))

CellData.get_domains(a::PIntegrands) = keys(a.dict)

function CellData.get_contribution(a::PIntegrands,meas::Measure)
  if haskey(a.dict,meas)
     return a.dict[meas]
  else
    @unreachable """\n
    There is no form associated with the given mesh in this PIntegrands object.
    """
  end
end

Base.getindex(a::PIntegrands,meas::Measure) = get_contribution(a,meas)

for op in (:+,:-)
  @eval begin
    function ($op)(a::PIntegrand,b::PIntegrand)
      c = PIntegrands()
      add_contribution!(c,a.meas,a)
      add_contribution!(c,b.meas,b,$op)
      c
    end

    function ($op)(a::PIntegrands,b::PIntegrand)
      c = copy(a)
      add_contribution!(c,b.meas,b,$op)
      c
    end

    ($op)(a::PIntegrand,b::PIntegrands) = $op(b,a)

    function ($op)(a::PIntegrands,b::PIntegrands)
      c = copy(a)
      for meas in get_domain(b)
        add_contribution!(c,meas,b[meas],$op)
      end
      c
    end
  end
end

function CellData.add_contribution!(a::PIntegrands,meas::Measure,b::PIntegrand,op=+)
  @assert !haskey(a.dict,meas)
  newobj = (x...) -> op(b.object(x...))
  a.dict[meas] = PIntegrand(newobj,meas)
  a
end

function Arrays.evaluate(ints::PIntegrands,inputs)
  conts = DomainContribution()
  for meas in get_domains(ints)
    int = ints[meas]
    conts += evaluate(int,inputs)
  end
  conts
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

  b = map(f->change_domain(f,quad.trian,quad.data_domain_style),fields)
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
