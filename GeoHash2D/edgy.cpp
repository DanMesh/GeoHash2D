//
//  edgy.cpp
//  LiveTracker
//
//  Created by Daniel Mesham on 07/08/2018.
//  Copyright © 2018 Daniel Mesham. All rights reserved.
//

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "edgy.hpp"

using namespace std;
using namespace cv;

Mat edgy::edgeToPointsMat(Vec4i edgeIn) {
    Vec4f edge = Vec4f(edgeIn);
    Mat ret1 = Mat( { edge[0], edge[1] } );
    Mat ret2 = Mat( { edge[2], edge[3] } );
    Mat ret;
    hconcat(ret1, ret2, ret);
    return ret;
}

float edgy::edgeAngle(Vec4i edge) {
    // Returns the angle in radians of the edge from the x axis towards the y axis
    // i.e. clockwise from horizontal right in DEGREES
    return fmod(cvFastArctan(edge[3] - edge[1], edge[2] - edge[0]), 180);
}

Point2f edgy::edgeCentre(Vec4i edge) {
    return Point2f( 0.5*(edge[0] + edge[2]) , 0.5*(edge[1] + edge[3]) );
}

float edgy::edgeLength(Vec4i edge) {
    return sqrt( pow(edge[2] - edge[0], 2) + pow(edge[3] - edge[1], 2) );
}

Point2i edgy::edgeEndpoint(Vec4i edge, int index) {
    return Point2i(edge[2*index] , edge[1 + 2*index]);
}

Vec2f edgy::edgeToRhoTheta(Vec4i edge) {
    Vec2f ret;
    
    if (edge[0] == edge[2]) {
        // Line is vertical
        ret[0] = edge[0];
        ret[1] = 0;
        return ret;
    }
    
    float m = (edge[3] - edge[1]) / (edge[2] - edge[0]);
    ret[0] = abs(edge[1] - m*edge[0]) / sqrt(m*m + 1);
    ret[1] = abs(edgeAngle(edge) - 90);
    return ret;
}

// TODO:
//  - add sort sides (see orange.cpp)
//  - create correpondences based on average lengths of opposite sides

vector<Vec4i> edgy::sortSides(vector<Vec4i> sides) {
    
    if (sides.size() < 4) return sides;
    
    for (int i = 0; i < sides.size()-1; i++) {
        
        Vec4i s = sides[i];
        vector<int> closest = {i+1, 0};
        double minDist = 10000;
        
        for (int j = i+1; j < sides.size(); j++) {  // For each other edge
            for (int k = 0; k < 2; k++) {           // For each endpoint
                double dist = norm(edgy::edgeEndpoint(s, 1) - edgy::edgeEndpoint(sides[j], k));
                if (dist < minDist) {
                    minDist = dist;
                    closest = {j, k};
                }
            }
        }
        
        // Move the closest side to be immediately after the current side
        if (closest[0] != i+1) {
            Vec4i tmp = sides[i+1];
            sides[i+1] = sides[closest[0]];
            sides[closest[0]] = tmp;
        }
        // Move the closest point to be the first in the side
        if (closest[1] != 0) {
            Point2i tmp = edgy::edgeEndpoint(sides[i+1], 0);
            sides[i+1][0] = sides[i+1][2];
            sides[i+1][1] = sides[i+1][3];
            sides[i+1][2] = tmp.x;
            sides[i+1][3] = tmp.y;
        }
    }
    return sides;
}

vector<Point2f> edgy::clusterEdges(vector<Point2f> edgeEnds, int k) {
    vector<Point2f> newEnds;
    vector<int> labels;
    int attempts = 5;
    double eps = 0.001;
    double compactness = kmeans(edgeEnds, k, labels,
                                TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, attempts, eps),
                                attempts, KMEANS_PP_CENTERS, newEnds);
    return newEnds;
}
