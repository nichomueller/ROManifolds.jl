// Gmsh project created on Fri Mar 20 14:01:03 2020
SetFactory("OpenCASCADE");

// Define Geometry
Rectangle(1) = {0, 0, 0, 2.5, 0.41, 0};
Disk(2) = {0.2, 0.2, 0, 0.05, 0.05};
BooleanDifference{ Surface{1}; Delete; }{ Surface{2}; }
Coherence;

// Define mesh sizes

Transfinite Curve {5} = 40 Using Progression 1;   // circle
Transfinite Curve {6,9} = 80 Using Progression 1; // walls
Transfinite Curve {7} = 25 Using Progression 1;   // inlet
Transfinite Curve {8} = 10 Using Progression 1;   // outlet

// Define Physical groups
Physical Surface("fluid") = {1};

Physical Curve("cylinder") = {5};
Physical Curve("walls")  = {6,9};
Physical Curve("inlet")  = {7};
Physical Curve("outlet") = {8};

Physical Point("inlet") = {6,8};
Physical Point("outlet") = {7,9};
