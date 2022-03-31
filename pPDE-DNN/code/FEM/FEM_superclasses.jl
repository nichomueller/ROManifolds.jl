abstract type FEM_problem

end


abstract type unsteady_problem <: FEM_problem

end


abstract type steady_problem <: unsteady_problem

end


abstract type FSI_problem <: unsteady_problem

end


abstract type Navier_Stokes_problem <: FSI_problem

end


abstract type Stokes_problem <: Navier_Stokes_problem

end


abstract type ADR_problem <: unsteady_problem

end


abstract type diffusion_problem <: ADR_problem

end


abstract type conservation_law_problem <: ADR_problem

end

#= 
#abstract steady_FSI_problem <: steady_problem

end


abstract type Navier_Stokes_problem <: FSI_problem

end


abstract type Stokes_problem <: Navier_Stokes_problem

end


abstract type ADR_problem <: unsteady_problem

end


abstract type diffusion_problem <: ADR_problem

end


abstract type Poisson_problem <: ADR_problem

end =#





