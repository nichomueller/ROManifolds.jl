abstract type FEMProblem end

struct UnsteadyProblem <: FEMProblem  end

struct SteadyProblem <: FEMProblem end

struct FSIProblem <: FEMProblem end

struct NavierStokesProblem <: FEMProblem end

struct StokesProblem <: FEMProblem end

struct ADRProblem <: FEMProblem end

struct DiffusionProblem <: FEMProblem end

struct ConservationLawProblem <: FEMProblem end

struct PoissonProblem <: FEMProblem end


struct FESpacePoisson <: FEMProblem
  Qₕ
  V₀
  V
  ϕᵥ
  ϕᵤ
  σₖ
  Nₕ
  dΩ
  dΓ
end


#=
#struct steady_FSI_problem <: steady_problem

end


struct Navier_Stokes_problem <: FSI_problem

end


struct Stokes_problem <: Navier_Stokes_problem

end


struct ADR_problem <: unsteady_problem

end


struct diffusion_problem <: ADR_problem

end


struct Poisson_problem <: ADR_problem

end =#
