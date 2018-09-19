//
//  edgy.hpp
//  LiveTracker
//
//  Created by Daniel Mesham on 07/08/2018.
//  Copyright © 2018 Daniel Mesham. All rights reserved.
//

#ifndef edgy_hpp
#define edgy_hpp

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

class edgy {
public:
    static Mat edgeToPointsMat(Vec4i edge);
    static float edgeAngle(Vec4i edge);
    static Point2f edgeCentre(Vec4i edge);
    static float edgeLength(Vec4i edge);
    static Point2i edgeEndpoint(Vec4i edge, int index);
    static Vec2f edgeToRhoTheta(Vec4i edge);
    static vector<Vec4i> sortSides(vector<Vec4i> sides);
    static vector<Point2f> clusterEdges(vector<Point2f> edgeEnds, int k);
    
};

#endif /* edgy_hpp */
