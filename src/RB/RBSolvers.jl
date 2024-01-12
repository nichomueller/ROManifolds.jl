struct RBSolver{I} <: LinearSolver
  info::I
end

struct RBSymbolicSetup <: SymbolicSetup
  solver::RBSolver
end

function Algebra.symbolic_setup(solver::RBSolver,mat::AbstractMatrix)
  RBSymbolicSetup(solver)
end

struct RBNumericalSetup <: NumericalSetup
  solver::RBSolver
end

function Algebra.numerical_setup(ss::RBSymbolicSetup,mat::AbstractMatrix)

end
