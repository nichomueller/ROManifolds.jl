struct RBAffineContribution
  dict::IdDict{RBIntegrationDomain,RBAffineDecomposition}
end

RBAffineContribution() = RBAffineContribution(
  IdDict{RBIntegrationDomain,RBAffineDecomposition}())

Base.copy(a::RBAffineContribution) = RBAffineContribution(copy(a.dict))

function (*)(a::Number,b::RBAffineContribution)
  c = RBAffineContribution()
  for (trian,array_old) in b.dict
    s = size(get_cell_map(trian))
    array_new = lazy_map(Broadcasting(*),Fill(a,s),array_old)
    add_contribution!(c,trian,array_new)
  end
  c
end

function integrate(f,b::GenericMeasure)
  c = integrate(f,b.quad)
  cont = DomainContribution()
  add_contribution!(cont,b.quad.trian,c)
  cont
end

struct TransientRBAffineContribution
  dict::IdDict{TransientRBIntegrationDomain,TransientRBAffineDecomposition}
end

TransientRBAffineContribution() = TransientRBAffineContribution(
  IdDict{TransientRBIntegrationDomain,TransientRBAffineDecomposition}())

Base.copy(a::TransientRBAffineContribution) = TransientRBAffineContribution(copy(a.dict))

function (*)(a::Number,b::TransientRBAffineContribution)
  c = RBAffineContribution()
  for (trian,array_old) in b.dict
    s = size(get_cell_map(trian))
    array_new = lazy_map(Broadcasting(*),Fill(a,s),array_old)
    add_contribution!(c,trian,array_new)
  end
  c
end

for (Tac,Tid,Tad) in zip((:RBAffineContribution,:TransientRBAffineContribution),
                    (:RBIntegrationDomain,:TransientRBIntegrationDomain),
                    (:RBAffineDecomposition,:TransientRBAffineDecomposition))

  @eval begin
    Gridap.CellData.num_domains(a::$Tac) = length(a.dict)

    Gridap.CellData.get_domains(a::$Tac) = keys(a.dict)

    function Gridap.CellData.get_contribution(a::$Tac,d::Tid)
      if haskey(a.dict,d)
         return a.dict[d]
      else
        @unreachable """\n
        There is not contribution associated with the given mesh in this Tac object.
        """
      end
    end

    Base.getindex(a::$Tac,i::$Tid) = get_contribution(a,i)

    function add_contribution!(a::$Tac,i::$Tid,d::$Tad,op=+)
      if haskey(a.dict,i)
        a.dict[i] = lazy_map(Broadcasting(op),a.dict[i],d)
      else
        if op == +
         a.dict[i] = d
        else
         a.dict[i] = lazy_map(Broadcasting(op),d)
        end
      end
      a
    end

    Base.sum(a::$Tac)= sum(map(sum,values(a.dict)))

    function (+)(a::$Tac,b::$Tac)
      c = copy(a)
      for (i,d) in b.dict
        add_contribution!(c,i,d)
      end
      c
    end

    function (-)(a::$Tac,b::$Tac)
      c = copy(a)
      for (i,d) in b.dict
        add_contribution!(c,i,d,-)
      end
      c
    end

    (*)(a::$Tac,b::Number) = b*a

    function get_array(a::$Tac)
      @assert num_domains(a) == 1 """\n
      Method get_array(a::$Tac) can be called only
      when the Tac object involves just one domain.
      """
      a.dict[first(keys(a.dict))]
    end

    function integrate(f,b::Measure)
      @abstractmethod
    end

    function get_cell_quadrature(b::Measure)
      @abstractmethod
    end

    function get_cell_points(a::Measure)
      quad = get_cell_quadrature(a)
      return get_cell_points(quad)
    end

    function (*)(a::Integrand,b::Measure)
      integrate(a.object,b)
    end

    (*)(b::Measure,a::Integrand) = a*b
  end

end
