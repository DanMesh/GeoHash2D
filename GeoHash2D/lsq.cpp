//
//  lsq.cpp
//  LiveTracker
//
//  Created by Daniel Mesham on 07/08/2018.
//  Copyright © 2018 Daniel Mesham. All rights reserved.
//

#include "lsq.hpp"

estimate lsq::poseEstimateLM(Vec6f pose1, Mat model, Mat target, Mat K, int maxIter) {
    // pose1: imitial pose parameters
    // model: model points in full homogeneous coords
    // target: image points, in 2D coords
    // K: intrinsic matrix
    // maxIter: max no of iterations, default if 0
    // rotOrder: order of rotations, default XYZ
    
    if (maxIter == 0) maxIter = MAX_ITERATIONS;
    
    Mat y = lsq::projection(pose1, model, K);
    float E = lsq::projectionError(target, y);
    
    int iterations = 0;
    while (E > ERROR_THRESHOLD && iterations < maxIter) {
        Mat J = lsq::jacobian(pose1, model, K);
        Mat eps;
        subtract(y.rowRange(0, 2).t(), target, eps);
        eps = lsq::pointsAsCol(eps.t());
        Mat Jp = J.t() * J;
        Jp = -Jp.inv() * J.t();
        Mat del = Jp * eps;
        
        Vec6f pose2 = pose1;
        for (int i = 0; i < 6; i++) {
            pose2[i] += del.at<float>(i);
        }
        
        y = lsq::projection(pose2, model, K);
        E = lsq::projectionError(target, y);
        
        pose1 = pose2;
        iterations++;
    }
    
    return estimate(pose1, E, iterations);
}

Mat lsq::translation(float x, float y, float z) {
    // Translate by the given x, y and z values
    float tmp[] = {x, y, z};
    return Mat(3, 1, CV_32FC1, tmp) * 1;
}

Mat lsq::rotation(float x, float y, float z) {
    // Rotate about the x, y then z axes with the given angles in radians
    float rotX[3][3] = {
        { 1,       0,       0 },
        { 0,  cos(x), -sin(x) },
        { 0,  sin(x),  cos(x) }
    };
    float rotY[3][3] = {
        {  cos(y),   0,  sin(y) },
        {       0,   1,       0 },
        { -sin(y),   0,  cos(y) }
    };
    float rotZ[3][3] = {
        {  cos(z), -sin(z),  0 },
        {  sin(z),  cos(z),  0 },
        {       0,       0,  1 }
    };
    
    Mat rX = Mat(3, 3, CV_32FC1, rotX);
    Mat rY = Mat(3, 3, CV_32FC1, rotY);
    Mat rZ = Mat(3, 3, CV_32FC1, rotZ);
    
    return rZ * rY * rX;
}

Mat lsq::projection(Vec6f pose, Mat model, Mat K) {
    Mat P;
    hconcat( rotation(pose[3], pose[4], pose[5]) , translation(pose[0], pose[1], pose[2]) , P);
    Mat y = (K * P) * model;
    
    Mat z = y.row(2);
    Mat norm;
    vconcat(z, z, norm);
    vconcat(norm, z, norm);
    divide(y, norm, y);
    return y;
}

float lsq::projectionError(Mat target, Mat proj) {
    transpose(proj.rowRange(0, 2), proj);
    
    Mat e;
    subtract(target, proj, e);
    multiply(e, e, e);
    reduce(e, e, 1, CV_REDUCE_SUM, CV_32FC1);
    sqrt(e, e);
    Mat eT;
    transpose(e, eT);
    e = eT * e;
    return e.at<float>(0);
}

Mat lsq::pointsAsCol(Mat points) {
    // Converts a matrix of points (each a column vector in homogeneous coordinates) into a single column of coordinates in standard coordinates (i.e. the last coordinate is removed)
    points = points.rowRange(0, 2);
    points = points.t();
    points = points.reshape(0, 1).t();
    return points;
}

Mat lsq::jacobian(Vec6f pose, Mat model, Mat K) {
    // Calculates the Jacobian for the given pose of model x
    Mat J = Mat(2*model.cols, 0, CV_32FC1);
    
    float dt = 1;
    float dr = CV_PI/180;
    vector<float> delta = {dt, dt, dt, dr, dr, dr};
    
    for (int i = 0; i < 6; i++) {
        Vec6f p1 = pose;
        p1[i] += delta[i];
        Vec6f p2 = pose;
        p2[i] -= delta[i];
        Mat j = (projection(p1, model, K) - projection(p2, model, K))/delta[i];
        hconcat(J, pointsAsCol(j), J);
    }
    
    return J;
}

Vec6f estimate::standardisePose(Vec6f pose) {
    // Ensures all angles are in (-PI,PI]
    for (int i = 3; i < 6; i++) {
        while (pose[i] > CV_PI)     pose[i] -= CV_2PI;
        while (pose[i] <= -CV_PI)   pose[i] += CV_2PI;
    }
    return pose;
}

bool estimate::mostSimilar(Vec6f poseA, Vec6f poseB, float alpha) {
    // Returns true if poseA is more similar to this pose than poseB
    // 'alpha' scales the error in the rotation (due to different units)
    Mat diff;
    hconcat(Mat(poseA - this->pose), Mat(poseB - this->pose), diff);
    Mat tra = diff.rowRange(0, 3);
    Mat rot = diff.rowRange(3, 6);
        
    tra = tra.t()*tra;
    float traDiff = tra.at<float>(1,1) - tra.at<float>(0,0);
    
    rot = rot.t()*rot;
    float rotDiff = rot.at<float>(1,1) - rot.at<float>(0,0);
    
    return (traDiff + alpha * rotDiff) >= 0;
}


// * * * * * * * * * * * * * * * * *
//   2D Methods
// * * * * * * * * * * * * * * * * *

estimate lsq::poseEstimate2D(Vec3f pose1, Mat model, Mat target, int maxIter) {
    // pose1: imitial pose parameters
    // model: model points in full 2D homogeneous coords (as columns)
    // target: image points in full 2D homogeneous coords  (as columns)
    // maxIter: max no of iterations, default if 0
    
    if (maxIter == 0) maxIter = MAX_ITERATIONS;
    
    Mat y = lsq::projection2D(pose1, model);
    float E = lsq::projectionError2D(target, y);
    
    int iterations = 0;
    while (E > ERROR_THRESHOLD && iterations < maxIter) {
        Mat J = lsq::jacobian2D(pose1, model);
        Mat eps;
        subtract(y, target, eps);
        eps = lsq::pointsAsCol(eps);
        Mat Jp = J.t() * J;
        Jp = -Jp.inv() * J.t();
        Mat del = Jp * eps;
        
        Vec3f pose2 = pose1;
        for (int i = 0; i < 3; i++) {
            pose2[i] += del.at<float>(i);
        }
        
        y = lsq::projection2D(pose2, model);
        E = lsq::projectionError2D(target, y);
        
        pose1 = pose2;
        iterations++;
    }
    
    Vec6f poseOut;
    poseOut[0] = pose1[0];
    poseOut[1] = pose1[1];
    poseOut[5] = pose1[2];
    
    return estimate(poseOut, E, iterations);
}

Mat lsq::projection2D(Vec3f pose, Mat model) {
    float a = pose[2];
    float proj[3][3] = {
        {  cos(a), -sin(a),  pose[0] },
        {  sin(a),  cos(a),  pose[1] },
        {       0,       0,    1     }
    };
    Mat P = Mat(3, 3, CV_32FC1, proj);
    
    return P * model;
}

float lsq::projectionError2D(Mat target, Mat proj) {
    Mat e;
    subtract(target, proj, e);
    multiply(e, e, e);
    reduce(e, e, 1, CV_REDUCE_SUM, CV_32FC1);
    sqrt(e, e);
    Mat eT;
    transpose(e, eT);
    e = eT * e;
    return e.at<float>(0);
}

Mat lsq::jacobian2D(Vec3f pose, Mat model) {
    // Calculates the Jacobian for the given pose of the model points
    Mat J = Mat(2*model.cols, 0, CV_32FC1);
    
    float dt = 1;
    float dr = CV_PI/180;
    vector<float> delta = {dt, dt, dr};
    
    for (int i = 0; i < 3; i++) {
        Vec3f p1 = pose;
        p1[i] += delta[i];
        Vec3f p2 = pose;
        p2[i] -= delta[i];
        Mat j = (projection2D(p1, model) - projection2D(p2, model))/delta[i];
        hconcat(J, pointsAsCol(j), J);
    }
    
    return J;
}
