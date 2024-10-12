// Gmsh project created on Sat Jan 28 16:13:18 2023
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {1, 0, 0, 1.0};
//+
Point(3) = {1, 0.5, 0, 1.0};
//+
Point(4) = {0, 0.5, 0, 1.0};
//+
Line(1) = {4, 1};
//+
Line(2) = {1, 2};
//+
Line(3) = {2, 3};
//+
Line(4) = {3, 4};
//+
Point(5) = {0.25, 0.15, 0, 1.0};
//+
Point(6) = {0.25, 0.25, 0, 1.0};
//+
Point(7) = {0.25, 0.35, 0, 1.0};
//+
Point(8) = {0.75, 0.35, 0, 1.0};
//+
Point(9) = {0.75, 0.25, 0, 1.0};
//+
Point(10) = {0.75, 0.15, 0, 1.0};
//+
Point(11) = {0.15, 0.25, 0, 1.0};
//+
Point(12) = {0.35, 0.25, 0, 1.0};
//+
Point(13) = {0.65, 0.25, 0, 1.0};
//+
Point(14) = {0.85, 0.25, 0, 1.0};
//+
Circle(5) = {11, 6, 5};
//+
Circle(6) = {5, 6, 12};
//+
Circle(7) = {12, 6, 7};
//+
Circle(8) = {7, 6, 11};
//+
Circle(9) = {10, 9, 14};
//+
Circle(10) = {14, 9, 8};
//+
Circle(11) = {8, 9, 13};
//+
Circle(12) = {13, 9, 10};
//+
Curve Loop(1) = {4, 1, 2, 3};
//+
Curve Loop(2) = {11, 12, 9, 10};
//+
Curve Loop(3) = {7, 8, 5, 6};
//+
Plane Surface(1) = {1, 2, 3};
//+
Extrude {0, 0, 0.15} {
  Surface{1}; 
}
//+
Physical Volume("domain") = {1};
//+
Physical Surface("inlet") = {33};
//+
Physical Surface("cylinder1") = {65, 61, 73, 69};
//+
Physical Surface("cylinder2") = {53, 49, 45, 57};
//+
Physical Curve("cylinder1_c") = {8, 5, 6, 7, 59, 64, 23, 22, 68, 25, 24, 60};
//+
Physical Curve("cylinder2_c") = {21, 11, 52, 12, 20, 44, 10, 18, 43, 19, 9, 48};
//+
Physical Curve("inlet_c") = {15, 28, 1, 32};
//+
Physical Point("inlet_p") = {16, 4, 1, 20};
//+
Physical Point("cylinder1_p") = {58, 53, 51, 63, 7, 11, 5, 12};
//+
Physical Point("cylinder2_p") = {31, 8, 43, 13, 38, 10, 33, 14};
//+
Physical Surface("sides") = {37, 74, 29, 1};
//+
Physical Curve("sides_c") = {4, 14, 2, 16, 1, 28, 15, 32};
//+
Physical Point("sides_p") = {4, 16, 20, 1};
