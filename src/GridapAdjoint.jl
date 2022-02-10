# The general rule for computing the adjoint

function Adjoint(ϕ,u,du,op,res,bg_params)#,solver)

  A = Gridap.jacobian(op,u) 
  Aᵀ = adjoint(A) 
  V = op.test
  λₕ = FEFunction(V,Aᵀ\du) #,solver)

  function Rλ(ϕₘ)
      ϕ = collect1d(ϕₘ)
      geo_params, _ = get_geo_params(ϕ,bg_params)
      sum(res(u,λₕ,geo_params))
  end

  ϕₘ =  reshape(ϕ,(length(ϕ),1))
  dϕₘ = ReverseDiff.gradient(Rλ,ϕₘ)
  dϕ = -collect1d(dϕₘ)

end

function u_to_j(u,ϕ,bg_params)

  weak_form_objects , geo_objects = get_geo_params(ϕ,bg_params)

  (V,U) = get_FE_Spaces(geo_objects)
  uₕ = FEFunction(V,u)

  jp = get_objective(uₕ,ϕ,weak_form_objects)

end

function ChainRulesCore.rrule(::typeof(u_to_j),u,ϕ,bg_params)

  weak_form_objects , geo_objects = get_geo_params(ϕ,bg_params)

  (V,U) = get_FE_Spaces(geo_objects)
  uₕ = FEFunction(V,u)

  jp = get_objective(uₕ,ϕ,weak_form_objects)

  function u_to_j_pullback(dj)

    (dΩ1, dΩ2, dΓ, dΓg1, dΓg2,  n_Γ, n_Γg, κ1, κ2, β)  = weak_form_objects
    j1(u1) = ∫(u1)dΩ1
    j2(u2) = ∫(u2)dΩ2
    uₕ₁, uₕ₂ = uₕ
    dfdu1 = ∇(j1)(uₕ₁)
    dfdu2 = ∇(j2)(uₕ₂)
    V1,V2=V
    dfdu_vec1 = assemble_vector(dfdu1,V1)
    dfdu_vec2 = assemble_vector(dfdu2,V2)
    dfdu_vec = append!(dfdu_vec1,dfdu_vec2)

    # Eventually, we should be able to do the above 10 lines in only 2 lines : djdu = ∇(jp)(uₕ); djddu_vec = assemble_vector(djdu,V) 
    # but it will take some changes in the implementaion of appendedarrays in gridap (see https://github.com/gridap/Gridap.jl/issues/661).
    # This also avoids redefining j here ( as j1 and j2 )
    # Anyway, if the problem was not multifield, we would not have this issue

    function jq(ϕₘ)

      ϕ = collect1d(ϕₘ)
      _weak_form_objects , geo_objects = get_geo_params(ϕ,bg_params)
      J(uₕ,ϕ,_weak_form_objects)

    end

    ϕₘ =  reshape(ϕ,(length(ϕ),1))
    dϕₘ = ReverseDiff.gradient(jq,ϕₘ)
    dj_dϕ = collect1d(dϕₘ)
    
    djϕ = dj*dj_dϕ

    (  NoTangent(), dj*dfdu_vec, djϕ, NoTangent() )
  end

jp, u_to_j_pullback

end

# ϕ_to_u

function ϕ_to_u(ϕ,bg_params)

uₕ, op, res = get_state(ϕ,bg_params) # run simulation
uₕ.free_values

end

function ChainRulesCore.rrule(::typeof(ϕ_to_u),ϕ,bg_params)

uₕ, opᵤ,resᵤ = get_state(ϕ,bg_params) # run simulation

function ϕ_to_u_pullback(du)

	dϕ = Adjoint(ϕ,uₕ,du,opᵤ,resᵤ,bg_params)     
	( NoTangent(),dϕ, NoTangent() )

end

uₕ.free_values, ϕ_to_u_pullback
    
end

# outer chain

function ϕ_to_j(ϕ,bg_params)
  u  = ϕ_to_u(ϕ,bg_params)      
  j  = u_to_j(u,ϕ,bg_params)   
end

function ChainRulesCore.rrule(::typeof(ϕ_to_j), ϕ::AbstractVector,bg_params)
    
	u,u_pullback    = rrule(ϕ_to_u,ϕ,bg_params)
	j, j_pullback   = rrule(u_to_j,u,ϕ,bg_params)
    
	function ϕ_to_j_pullback(dj)
    
    _, du, dϕ₍ⱼ₎,_   = j_pullback(dj)
    _, dϕ₍ᵤ₎, _       = u_pullback(du)
        dϕ          = dϕ₍ᵤ₎ + dϕ₍ⱼ₎

    ( NoTangent(), dϕ, NoTangent() )
	end
    
	j, ϕ_to_j_pullback
end

