//
//  lsq.hpp
//  LiveTracker
//
//  Created by Daniel Mesham on 07/08/2018.
//  Copyright © 2018 Daniel Mesham. All rights reserved.
//

#ifndef lsq_hpp
#define lsq_hpp

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//      A class defining a least squares estimated pose
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
class estimate {
public:
    estimate( Vec6f pose_in, float error_in, float iter_in ) : pose(standardisePose(pose_in)), error(error_in), iterations(iter_in) {}
    estimate() : error(10000), iterations(100) {}
    void print() {cout << pose << "\nIterations = " << iterations << "\nError = " << error << "\n\n";}
    bool mostSimilar(Vec6f poseA, Vec6f poseB, float alpha = 10);
    
    bool operator < (const estimate& e) const
    {
        return (error < e.error);
    }
    
private:
    static Vec6f standardisePose(Vec6f pose);
    
public:
    Vec6f pose;
    float error, iterations;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//      A library of least squares methods
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
class lsq {
    
/*
    METHODS
 */
public:
    static estimate poseEstimateLM(Vec6f pose1, Mat x, Mat target, Mat K, int maxIter = MAX_ITERATIONS);
    static Mat translation(float x, float y, float z);
    static Mat rotation(float x, float y, float z);
    static Mat projection(Vec6f pose, Mat x, Mat K);
    static float projectionError(Mat target, Mat proj);
    static Mat pointsAsCol(Mat points);
    static Mat jacobian(Vec6f pose, Mat x, Mat K);
    
    // 2D methods for affine case
    static estimate poseEstimate2D(Vec3f pose1, Mat model, Mat target, int maxIter = MAX_ITERATIONS);
    static Mat projection2D(Vec3f pose, Mat model);
    static float projectionError2D(Mat target, Mat proj);
    static Mat jacobian2D(Vec3f pose, Mat model);
        
/*
    CONSTANTS
 */
public:
    static const int MAX_ITERATIONS = 20;
    static const float ERROR_THRESHOLD;

};




#endif /* lsq_hpp */
