// Gmsh project created on Fri Mar 20 14:01:03 2020
SetFactory("OpenCASCADE");

// Define Geometry
Rectangle(1) = {0, 0, 0, 2.5, 0.41, 0};
Disk(2) = {0.2, 0.2, 0, 0.05, 0.05};
BooleanDifference{ Surface{1}; Delete; }{ Surface{2}; }
Extrude {0,0,0.41} {
  Surface{1}; Layers{ 15 };
}
Coherence;

// Define mesh sizes

Transfinite Curve {5,19} = 40 Using Progression 1;   // circle
Transfinite Curve {6,9,12,17} = 60 Using Progression 1; // walls
Transfinite Curve {7,10,13,14} = 20 Using Progression 1;   // inlet
Transfinite Curve {8,11,15,16} = 10 Using Progression 1;   // outlet

// Define Physical groups
Physical Volume("fluid") = {1};

Physical Surface("cylinder") = {7};
Physical Surface("walls")  = {1,3,6,8};
Physical Surface("inlet")  = {4};
Physical Surface("outlet") = {5};

Physical Curve("cylinder") = {5,19};
Physical Curve("walls")  = {6,9,12,17};
Physical Curve("inlet")  = {7,10,13,14};
Physical Curve("outlet") = {8,11,15,16};

Physical Point("inlet") = {6,8,10,12};
Physical Point("outlet") = {7,9,11,13};
