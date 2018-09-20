//
//  main.cpp
//  GeoHash2D
//
//  Created by Daniel Mesham on 18/09/2018.
//  Copyright © 2018 Daniel Mesham. All rights reserved.
//

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio/videoio.hpp>

#include "area.hpp"
#include "asm.hpp"
#include "edgy.hpp"
#include "geo_hash.h"
#include "hashing.hpp"
#include "lsq.hpp"
#include "models.hpp"
#include "orange.hpp"

#include <chrono>
#include <iostream>

using namespace std;
using namespace cv;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//      Constants
// * * * * * * * * * * * * * * * * * * * * * * * * * * * *

// The intrinsic matrix: Mac webcam
static float intrinsicMatrix[3][3] = {
    { 1047.7,    0  , 548.1 },
    {    0  , 1049.2, 362.9 },
    {    0  ,    0  ,   1   }
};
static Mat K = Mat(3,3, CV_32FC1, intrinsicMatrix);

static float binWidth = 1;

static string dataFolder = "../../../../../data/";

static bool REPORT_ERRORS = true; // Whether to report the area error (slows performance)



// * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//      Main Method
// * * * * * * * * * * * * * * * * * * * * * * * * * * * *

int main(int argc, const char * argv[]) {
    
    // * * * * * * * * * * * * * * * * *
    //   MODEL CREATION
    // * * * * * * * * * * * * * * * * *
    
    Model * modelRect = new Rectangle(60, 80, Scalar(20, 65, 165));
    Model * modelDog = new Dog(Scalar(19, 89, 64));
    Model * modelArrow = new Arrow(Scalar(108, 79, 28));
    
    Model * calibModel = modelArrow;
    vector<Model *> model = {modelArrow, modelDog, modelRect};
    
    // * * * * * * * * * * * * * * * * *
    //   HASHING
    // * * * * * * * * * * * * * * * * *
    
    auto startHash = chrono::system_clock::now(); // Start hashing timer
    
    vector<vector<HashTable>> tables;
    
    for (int m = 0; m < model.size(); m++) {
        
        vector<HashTable> modelTables;
        
        vector<Point3f> modelPoints3D = model[m]->getVertices();
        
        // Convert 3D coordinates to 2D
        vector<Point2f> modelPoints2D;
        for (int i = 0; i < modelPoints3D.size(); i++) {
            modelPoints2D.push_back( Point2f(modelPoints3D[i].x, modelPoints3D[i].y) );
        }
        
        // A list of model basis pairs
        vector<vector<int>> basisList = model[m]->getEdgeBasisList();
        
        // Get the visibility mask (should be all true)
        vector<bool> vis = model[m]->visibilityMask(0, 0);
        
        for (int i = 0; i < basisList.size(); i++) {
            vector<int> basisIndex = basisList[i];
            
            // Don't make a hash table if the basis isn't visible
            if (!vis[basisIndex[0]] || !vis[basisIndex[1]]) continue;
            
            modelTables.push_back(hashing::createTable(basisIndex, modelPoints2D, vis, binWidth));
        }
        tables.push_back(modelTables);
    }
    
    auto endHash = chrono::system_clock::now();
    chrono::duration<double> timeHash = endHash-startHash;
    cout << "Hashing time = " << timeHash.count()*1000.0 << " ms" << endl;
    
    // * * * * * * * * * * * * * * * * *
    //   OPEN THE FIRST FRAME
    // * * * * * * * * * * * * * * * * *
    
    Mat frame;
    String filename = "Trio_1.avi";
    VideoCapture cap(dataFolder + filename);
    //VideoCapture cap(0); waitKey(1000);   // Uncomment this line to try live tracking
    if(!cap.isOpened()) return -1;
    
    cap >> frame;
    imshow("Frame", frame);
    
    // * * * * * * * * * * * * * * * * *
    //   FIND THE CAMERA POSE
    // * * * * * * * * * * * * * * * * *
    
    // Find the area & centoid of the object in the image
    Mat segInit = orange::segmentByColour(frame, calibModel->colour);
    cvtColor(segInit, segInit, CV_BGR2GRAY);
    threshold(segInit, segInit, 0, 255, CV_THRESH_BINARY);
    Point centroid = ASM::getCentroid(segInit);
    double area = ASM::getArea(segInit);
    
    // Draw the model at the default position and find the area & cetroid
    Vec6f initPose = {0, 0, 300, -CV_PI/4, 0, 0};
    Mat initGuess = Mat::zeros(frame.rows, frame.cols, frame.type());
    calibModel->draw(initGuess, initPose, K, false);
    cvtColor(initGuess, initGuess, CV_BGR2GRAY);
    threshold(initGuess, initGuess, 0, 255, CV_THRESH_BINARY);
    Point modelCentroid = ASM::getCentroid(initGuess);
    double modelArea = ASM::getArea(initGuess);
    
    // Convert centroids to 3D/homogeneous coordinates
    Mat centroid2D;
    hconcat( Mat(centroid), Mat(modelCentroid), centroid2D );
    vconcat(centroid2D, Mat::ones(1, 2, centroid2D.type()), centroid2D);
    centroid2D.convertTo(centroid2D, K.type());
    Mat centroid3D = K.inv() * centroid2D;
    
    // Estimate the depth from the ratio of the model and measured areas,
    // and create a pose guess from that.
    // Note that the x & y coordinates need to be calculated using the pose
    // of the centroid relative to the synthetic model image's centroid.
    double zGuess = initPose[2] * sqrt(modelArea/area);
    centroid3D *= zGuess;
    initPose[0] = centroid3D.at<float>(0, 0) - centroid3D.at<float>(0, 1);
    initPose[1] = centroid3D.at<float>(1, 0) - centroid3D.at<float>(1, 1);
    initPose[2] = zGuess;
    estimate initEst = estimate(initPose, 0, 0);
    
    // Detect edges
    Mat canny;
    Canny(segInit, canny, 70, 210);
    
    // Extract the image edge point coordinates
    Mat edges;
    findNonZero(canny, edges);
    
    int iterations = 1;
    double error = lsq::ERROR_THRESHOLD + 1;
    while (error > lsq::ERROR_THRESHOLD && iterations < 100) {
        // Generate a set of whiskers
        vector<Whisker> whiskers = ASM::projectToWhiskers(calibModel, initEst.pose, K);
        
        // Sample along the model edges and find the edges that intersect each whisker
        Mat targetPoints = Mat(2, 0, CV_32S);
        Mat whiskerModel = Mat(4, 0, CV_32FC1);
        for (int w = 0; w < whiskers.size(); w++) {
            Point closestEdge = whiskers[w].closestEdgePoint(edges);
            if (closestEdge == Point(-1,-1)) continue;
            hconcat(whiskerModel, whiskers[w].modelCentre, whiskerModel);
            hconcat(targetPoints, Mat(closestEdge), targetPoints);
        }
        
        targetPoints.convertTo(targetPoints, CV_32FC1);
        
        // Use least squares to match the sampled edges to each other
        initEst = lsq::poseEstimateLM(initEst.pose, whiskerModel, targetPoints.t(), K, 2);
        
        iterations++;
        //waitKey(0);
    }
    
    Mat frameCalib;
    frame.copyTo(frameCalib);
    calibModel->draw(frameCalib, initEst.pose, K);
    imshow("Calibration Model", frameCalib);
    
    Mat P;
    Mat R = lsq::rotation(initEst.pose[3], initEst.pose[4], initEst.pose[5]);
    Mat t = lsq::translation(initEst.pose[0], initEst.pose[1], initEst.pose[2]);
    hconcat(R.colRange(0, 2), t, P);
    P = (K * P);
    
    // The coordinates of the centre of the image/frame
    int Ox = frame.cols/2;
    int Oy = frame.rows/2;
    
    waitKey(0);
    
    
    // * * * * * * * * * * * * * * * * *
    //   INPUT LOOP
    // * * * * * * * * * * * * * * * * *
    
    vector<double> times = {};
    double longestTime = 0.0;
    vector<vector<double>> errors = vector<vector<double>>(model.size());
    vector<double> worstError = vector<double>(model.size());
    
    while (!frame.empty()) {
        
        // * * * * * * * * * * * * * * * * *
        //   ESTIMATE POSES OF ALL MODELS
        // * * * * * * * * * * * * * * * * *
        
        auto startRecog = chrono::system_clock::now(); // Start recognition timer
        
        for (int m = 0; m < model.size(); m++) {
            
            // Segment the image
            Mat seg = orange::segmentByColour(frame, model[m]->colour);
            cvtColor(seg, seg, CV_BGR2GRAY);
            threshold(seg, seg, 0, 255, CV_THRESH_BINARY);
            
            // Convert the segmented image to a list of foreground coordinates
            Mat target;
            findNonZero(seg, target);
            target = target.t();
            Mat targetRows[2];
            split(target, targetRows);
            vconcat(targetRows[0], targetRows[1], target);
            vconcat(target, Mat::ones(1, target.cols, target.type()), target);
            target.convertTo(target, P.type());
            
            // Convert image coords to plane coords
            Mat y = P.inv() * target;
            Mat z = y.row(2);
            Mat norm;
            vconcat(z, z, norm);
            vconcat(norm, z, norm);
            divide(y, norm, y);
            
            // Draw the 2D plane
            Mat plane = Mat::zeros(frame.rows, frame.cols, frame.type());
            
            for (int i = 0; i < y.cols; i++) {
                Point pt = Point(y.at<float>(0, i) + Ox, y.at<float>(1, i) + Oy);
                circle(plane, pt, 0, Scalar(0, 0, 255));
            }
            //imshow("plane", plane);
            
            // Find the edge lines in the 2D planes
            vector<Vec4i> lines = orange::borderLines(plane);
            vector<Point2f> imgPoints;
            
            // Convert the edges to a set of points
            for (int i = 0; i < lines.size(); i++) {
                Point p1 = Point(lines[i][0], lines[i][1]);
                Point p2 = Point(lines[i][2], lines[i][3]);
                
                Scalar colour = Scalar(0,255,0);
                if (i == 0) colour = Scalar(255,0,0);
                
                line(plane, p1, p2, colour);
                imgPoints.push_back(Point2f(p1));
                imgPoints.push_back(Point2f(p2));
            }
            //imshow("plane2", plane);
            
            // * * * * * * * * * * * * * *
            //      RECOGNITION
            // * * * * * * * * * * * * * *
            
            estimate bestEst = estimate({0,0,0,0,0,0}, 10000, 100);  // The best estimated pose so far
            int edge = 0;   // The detected edge to use as a basis
            
            while (edge < lines.size()) {
                vector<int> imgBasis = {2*edge, 2*edge +1};
                
                // Vote for the tables using the given basis
                vector<HashTable> votedTables = hashing::voteForTables(tables[m], imgPoints, imgBasis);
                int maxVotes = votedTables[0].votes;
                
                for (int i = 0; i < votedTables.size(); i++) {
                    HashTable t = votedTables[i];
                    if (t.votes < MIN(200, maxVotes)) break;
                    
                    vector<Mat> orderedPoints = hashing::getOrderedPoints(imgBasis, t, model[m]->getVertices(), imgPoints);
                    
                    Mat newModel = orderedPoints[0];
                    Mat newTarget = orderedPoints[1];
                    
                    // Require at least 4 correspondences
                    if (newModel.cols < 4) continue;
                    
                    vconcat(newModel.rowRange(0, 2), newModel.row(3), newModel);
                    vconcat(newTarget.t(), Mat::ones(1, newTarget.rows, newTarget.type()), newTarget);
                    
                    // Use least squares to estimate pose
                    estimate est = lsq::poseEstimate2D({0,0,0}, newModel, newTarget);
                    if (est.error < bestEst.error) bestEst = est;
                }
                edge++;
            }
            
            // * * * * * * * * * * * * * * * * *
            //      SHOW THE ESTIMATED POSE
            // * * * * * * * * * * * * * * * * *
            
            Mat modelMat = model[m]->pointsToMat();
            vconcat(modelMat.rowRange(0, 2), modelMat.row(3), modelMat);
            
            // Find the coordintes using the LSQ result
            Vec3f poseLSQ = {bestEst.pose[0] - Ox, bestEst.pose[1] - Oy, bestEst.pose[5]};
            Mat modelInPlane = lsq::projection2D(poseLSQ, modelMat);
            
            // Project to the camera
            y = P * modelInPlane;
            z = y.row(2);
            vconcat(z, z, norm);
            vconcat(norm, z, norm);
            divide(y, norm, y);
            
            // Draw the shape
            vector<Point> contour;
            for (int i = 0; i < y.cols; i++) {
                contour.push_back(Point(y.at<float>(0,i), y.at<float>(1,i)));
            }
            
            const Point *pts = (const cv::Point*) Mat(contour).data;
            int npts = Mat(contour).rows;
            
            polylines(frame, &pts, &npts, 1, true, Scalar(0, 255, 0));
            
            // Measure and report the unexplained area
            if (!REPORT_ERRORS) continue;
            Mat silhouette = Mat(frame.rows, frame.cols,  CV_8UC1, Scalar(0));
            fillPoly(silhouette, &pts, &npts, 1, Scalar(255));
            double unexplainedArea = area::unexplainedArea(silhouette, seg);
            errors[m].push_back(unexplainedArea);
            if (unexplainedArea > worstError[m]) worstError[m] = unexplainedArea;
        }
        imshow("Frame", frame);
        
        auto endRecog = chrono::system_clock::now();
        chrono::duration<double> timeRecog = endRecog-startRecog;
        double time = timeRecog.count()*1000.0;
        cout << "Recognition time = " << time << " ms" << endl;
        
        times.push_back(time);
        if (time > longestTime) longestTime = time;
        
        // Get next frame
        cap.grab();
        cap >> frame;
        
        if (waitKey(1) == 'q') break;
    }
    
    vector<double> meanTime, stdDevTime;
    meanStdDev(times, meanTime, stdDevTime);
    
    cout << endl << endl;
    cout << "No. frames   = " << times.size() << endl;
    cout << "Avg time     = " << meanTime[0] << " ms     " << 1000.0/meanTime[0] << " fps" << endl;
    cout << "stdDev time  = " << stdDevTime[0] << " ms" << endl;
    cout << "Longest time = " << longestTime << " ms     " << 1000.0/longestTime << " fps" << endl;
    
    // Report errors
    if (!REPORT_ERRORS) return 0;
    cout << endl << "AREA ERRORS:" << endl << "Model   Mean     StDev    Worst" << endl;
    for (int m = 0; m < model.size(); m++) {
        vector<double> meanError, stdDevError;
        meanStdDev(errors[m], meanError, stdDevError);
        printf("%4i    %5.2f    %5.2f    %5.2f \n", m, meanError[0], stdDevError[0], worstError[m]);
        //cout << m << "   " << meanError[0] << "   " << stdDevError[0] << "   " << worstError[m] << endl;
    }
    
    return 0;
}
