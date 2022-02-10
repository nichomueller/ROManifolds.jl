using Gridap
using Gridap.FESpaces
using Flux
using Gridap.ReferenceFEs
using GridapEmbedded.LevelSetCutters
using Gridap.ReferenceFEs
using Gridap.Arrays
using Gridap.MultiField
using Gridap.Algebra
using GridapEmbedded
using GridapEmbedded.LevelSetCutters
using ChainRulesCore
using ReverseDiff
using ChainRulesTestUtils

include("../src/GridapAdjoint.jl")

function get_bg_params()

	n_cells = 5
	n_holes = 4 
	αᵩ = 0.0
	L=1
	dimensions = 2 
	domain = (0,L,0,L)
	cells=(n_cells,n_cells)
	h=L/n_cells
	bgmodel = CartesianDiscreteModel(domain,cells)
	order_u = 1
	degree_u = 2 * order_u
	reffe_u = ReferenceFE(lagrangian,Float64,order_u)
	Vbg = TestFESpace(bgmodel,FiniteElements(PhysicalDomain(),bgmodel,lagrangian,Float64,order_u))
	Ωbg = Triangulation(bgmodel)
	dΩ_bg = Measure(Ωbg,degree_u)
	point_to_coords = collect1d(get_node_coordinates(bgmodel))
	
	bg_params = Dict{Symbol,Any}(

	  :αᵩ=>αᵩ,
	  :L=>L,
	  :dimensions=>dimensions,
	  :domain=>domain,
	  :n_cells=>n_cells,
	  :cells=>cells,
	  :h=>h,
	  :bgmodel=>bgmodel,
	  :order_u=>order_u,
	  :reffe_u=>reffe_u,
	  :Vbg=>Vbg,
	  :Ωbg=>Ωbg,
	  :dΩ_bg=>dΩ_bg,
	  :point_to_coords=>point_to_coords,

	)
	
	bg_params
	
end

function get_geo_params(ϕ,bg_params)

	bgmodel = bg_params[:bgmodel]
	point_to_coords = bg_params[:point_to_coords]
	Ωbg = bg_params[:Ωbg]

  # Material Parameters and loads
	α1 = 1
	α2 = 1e-2

	geo1 = DiscreteGeometry(ϕ,point_to_coords,name="")
	geo2 = DiscreteGeometry(-ϕ,point_to_coords,name="")

	cutgeo1 = cut(bgmodel,geo1)
	cutgeo2 = cut(bgmodel,geo2)

	Γ = EmbeddedBoundary(cutgeo1)

	# Setup interpolation meshes
	Ω1_act = Triangulation(cutgeo1,ACTIVE)
	Ω2_act = Triangulation(cutgeo2,ACTIVE)

	# Setup integration meshes
	Ω1 = Triangulation(cutgeo1,PHYSICAL)
	Ω2 = Triangulation(cutgeo2,PHYSICAL)

	Γg1 = GhostSkeleton(cutgeo1)
	Γg2 = GhostSkeleton(cutgeo2)

	# Setup Lebesgue measures
	order = 1
	degree = 2*order
	dΩ1 = Measure(Ω1,degree)
	dΩ2 = Measure(Ω2,degree)
	dΓ = Measure(Γ,degree)
	dΓg1 = Measure(Γg1,degree)
	dΓg2 = Measure(Γg2,degree)
	n_Γ = get_normal_vector(Γ)
	n_Γg1 = get_normal_vector(Γg1)
	n_Γg2 = get_normal_vector(Γg2)

	# Setup stabilization parameters
	meas_K1 = get_cell_measure(Ω1, Ωbg)
	meas_K2 = get_cell_measure(Ω2, Ωbg)
	meas_KΓ = get_cell_measure(Γ, Ωbg)
	
	γ_hat = 2
	κ1 = CellField( (α2*meas_K1) ./ (α2*meas_K1 .+ α1*meas_K2), Ωbg)
	κ2 = CellField( (α1*meas_K2) ./ (α2*meas_K1 .+ α1*meas_K2), Ωbg)
	β = CellField( (γ_hat*meas_KΓ) ./ ( meas_K1/α1 .+ meas_K2/α2 ), Ωbg)

	( (dΩ1, dΩ2, dΓ, dΓg1 ,dΓg2 , n_Γ, n_Γg1, n_Γg2, κ1, κ2, β), (cutgeo1,cutgeo2,order,α1,α2, Ω1,Ω2,Ω1_act,Ω2_act) )

end

function get_FE_Spaces(geo_objects)

  (cutgeo1,cutgeo2,order,α1,α2, Ω1,Ω2,Ω1_act,Ω2_act) = geo_objects

	# Spaces 
	ud(x) = 0.0

	V1 = TestFESpace(
		Ω1_act,
		ReferenceFE(lagrangian,Float64,order),
		conformity=:H1,
		dirichlet_tags=[1,3,4,6,7]
		)

	V2 = TestFESpace(
		Ω2_act,
		ReferenceFE(lagrangian,Float64,order),
		conformity=:H1, 
		dirichlet_tags=[1,3,4,6,7]
	)

	U1 = TrialFESpace(V1,ud)
	U2 = TrialFESpace(V2,ud)

	V = MultiFieldFESpace([V1,V2])
	U = MultiFieldFESpace([U1,U2])

	(V,U)

end

# u_to_j

function get_objective(uₕ,ϕ,weak_form_objects)

  (dΩ1, dΩ2, dΓ, dΓg1, dΓg2,  n_Γ, n_Γg, κ1, κ2, β)  = weak_form_objects
  uₕ₁, uₕ₂ = uₕ
  jp = sum( ∫(uₕ₁)dΩ1 + ∫(uₕ₂)dΩ2 )

end

# ϕ_to_u

function get_state(ϕ,bg_params)

	h = bg_params[:h]	

	( (dΩ1, dΩ2, dΓ, dΓg1 ,dΓg2 , n_Γ, n_Γg1, n_Γg2, κ1, κ2, β), (cutgeo1,cutgeo2,order,α1,α2, Ω1,Ω2,Ω1_act,Ω2_act) ) = get_geo_params(ϕ,bg_params)

	(V,U) = get_FE_Spaces((cutgeo1,cutgeo2,order,α1,α2, Ω1,Ω2,Ω1_act,Ω2_act))

	# Weak form
	jump_u(u1,u2) = u1 - u2
	mean_q(u1,u2,κ1,κ2) = κ1*α1*∇(u1) + κ2*α2*∇(u2)
	mean_u(u1,u2,κ1,κ2) = κ2*u1 + κ1*u2

	f1(x) = 1e-2
	f2(x) = 1e-2

	fϕ = (dΩ1, dΩ2, dΓ, dΓg1 ,dΓg2 , n_Γ, n_Γg1, n_Γg2, κ1, κ2, β)

	γg = 0.1

	a( (u1,u2), (v1,v2),
	(dΩ1, dΩ2, dΓ, dΓg1 ,dΓg2 , n_Γ, n_Γg1, n_Γg2, κ1, κ2, β)) =
	∫( α1*∇(v1)⋅∇(u1) ) * dΩ1 + ∫( α2* ∇(v2)⋅∇(u2) ) * dΩ2 + 
	∫( β*jump_u(v1,v2)*jump_u(u1,u2)     )dΓ +
	∫( - n_Γ⋅mean_q(u1,u2,κ1,κ2)*jump_u(v1,v2) )dΓ + 
	∫( - n_Γ⋅mean_q(v1,v2,κ1,κ2)*jump_u(u1,u2) )dΓ + 
	∫( (γg*h)*jump(n_Γg1⋅∇(v1))*jump(n_Γg1⋅∇(u1)) ) * dΓg1 + 
	∫( (γg*h)*jump(n_Γg2⋅∇(v2))*jump(n_Γg2⋅∇(u2)) ) * dΓg2

	l( (v1,v2),
	(dΩ1, dΩ2, dΓ, dΓg1 ,dΓg2 , n_Γ, n_Γg1, n_Γg2, κ1, κ2, β) ) =
	∫( v1*f1 )dΩ1 + 
	∫( v2*f2 )dΩ2 

	res(u,v,fϕ) = a(u,v,fϕ) - l(v,fϕ) 

	a(fϕ) = (u,v) -> a(u,v,fϕ)
	l(fϕ) = v -> l(v,fϕ)

	op = AffineFEOperator(a(fϕ),l(fϕ),U,V)
	uh = Gridap.solve(op)

	#writevtk(Ω1,"Ω1_",cellfields=["uh"=>uh[1]])
	#writevtk(Ω2,"Ω2_",cellfields=["uh"=>uh[2]])

	(uh,op,res)

end

bg_params = get_bg_params(n_cells)

# generate initial guess
R = 0.7
geo1 = disk(R,)
bgmodel=bg_params[:bgmodel]
cutgeo = cut(bgmodel,geo1)
geoc = discretize(geo1,bgmodel)
ϕ₀ = geoc.tree.data[1]

j,djdp =  rrule(ϕ_to_j,ϕ₀,bg_params)
djdp(1)

#test_rrule(ϕ_to_j,ϕ₀,bg_params)