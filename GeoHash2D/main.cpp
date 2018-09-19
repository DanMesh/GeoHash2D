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
    
    Model * model = modelArrow;
    
    // * * * * * * * * * * * * * * * * *
    //   HASHING
    // * * * * * * * * * * * * * * * * *
    
    auto startHash = chrono::system_clock::now(); // Start hashing timer
    
    vector<HashTable> tables;
    
    vector<Point3f> modelPoints3D = model->getVertices();
    
    // Convert 3D coordinates to 2D
    vector<Point2f> modelPoints2D;
    for (int i = 0; i < modelPoints3D.size(); i++) {
        modelPoints2D.push_back( Point2f(modelPoints3D[i].x, modelPoints3D[i].y) );
    }
    
    // A list of model basis pairs
    vector<vector<int>> basisList = model->getEdgeBasisList();
    
    // Get the visibility mask (should be all true)
    vector<bool> vis = model->visibilityMask(0, 0);
    
    for (int i = 0; i < basisList.size(); i++) {
        vector<int> basisIndex = basisList[i];
        
        // Don't make a hash table if the basis isn't visible
        if (!vis[basisIndex[0]] || !vis[basisIndex[1]]) continue;
        
        tables.push_back(hashing::createTable(basisIndex, modelPoints2D, vis, binWidth));
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
    Mat segInit = orange::segmentByColour(frame, model->colour);
    cvtColor(segInit, segInit, CV_BGR2GRAY);
    threshold(segInit, segInit, 0, 255, CV_THRESH_BINARY);
    Point centroid = ASM::getCentroid(segInit);
    double area = ASM::getArea(segInit);
    
    // Draw the model at the default position and find the area & cetroid
    Vec6f initPose = {0, 0, 300, -CV_PI/4, 0, 0};
    Mat initGuess = Mat::zeros(frame.rows, frame.cols, frame.type());
    model->draw(initGuess, initPose, K, false);
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
        vector<Whisker> whiskers = ASM::projectToWhiskers(model, initEst.pose, K);
        
        Mat cannyTest;
        canny.copyTo(cannyTest);
        
        // Sample along the model edges and find the edges that intersect each whisker
        Mat targetPoints = Mat(2, 0, CV_32S);
        Mat whiskerModel = Mat(4, 0, CV_32FC1);
        for (int w = 0; w < whiskers.size(); w++) {
            Point closestEdge = whiskers[w].closestEdgePoint(edges);
            if (closestEdge == Point(-1,-1)) continue;
            hconcat(whiskerModel, whiskers[w].modelCentre, whiskerModel);
            hconcat(targetPoints, Mat(closestEdge), targetPoints);
            
            // TRACE: Display the whiskers
            circle(cannyTest, closestEdge, 3, Scalar(120));
            circle(cannyTest, whiskers[w].centre, 3, Scalar(255));
            line(cannyTest, closestEdge, whiskers[w].centre, Scalar(150));
        }
        imshow("CannyTest", cannyTest);
        
        targetPoints.convertTo(targetPoints, CV_32FC1);
        
        // Use least squares to match the sampled edges to each other
        initEst = lsq::poseEstimateLM(initEst.pose, whiskerModel, targetPoints.t(), K, 2);
        
        iterations++;
        //waitKey(0);
    }
    cout << "Iterations = " << iterations << endl;
    cout << "Error = " << error << endl;
    
    model->draw(frame, initEst.pose, K);
    imshow("Frame", frame);
    
    Mat P;
    Mat R = lsq::rotation(initEst.pose[3], initEst.pose[4], initEst.pose[5]);
    Mat t = lsq::translation(initEst.pose[0], initEst.pose[1], initEst.pose[2]);
    hconcat(R.colRange(0, 2), t, P);
    P = (K * P);
    
    // * * * * * * * * * * * * * * * * *
    //   TRY VISUALISE THE HOMOGRAPHY
    // * * * * * * * * * * * * * * * * *
    
    // Convert the segmented image to a list of foreground coordinates
    Mat target;
    findNonZero(segInit, target);
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
    int Ox = frame.cols/2;
    int Oy = frame.rows/2;
    for (int i = 0; i < y.cols; i++) {
        Point pt = Point(y.at<float>(0, i) + Ox, y.at<float>(1, i) + Oy);
        circle(plane, pt, 0, Scalar(0, 0, 255));
    }
    imshow("plane", plane);
    
    // Find the edge lines in the 2D planes
    vector<Vec4i> lines = orange::borderLines(plane);
    vector<Point2f> imgPoints;
    
    for (int i = 0; i < lines.size(); i++) {
        Point p1 = Point(lines[i][0], lines[i][1]);
        Point p2 = Point(lines[i][2], lines[i][3]);
        line(plane, p1, p2, Scalar(0,255,0));
        imgPoints.push_back(Point2f(p1));
        imgPoints.push_back(Point2f(p2));
    }
    imshow("plane2", plane);
    
    
    // * * * * * * * * * * * * * *
    //      RECOGNITION
    // * * * * * * * * * * * * * *
    
    auto startRecog = chrono::system_clock::now(); // Start recognition timer
    
    Mat H;          // The best homography thus far
    int edge = 0;   // The detected edge to use as a basis
    
    while (edge < lines.size() && edge < 2) {
        //TRACE:
        cout << endl << "TRYING EDGE #" << edge << endl;
        
        vector<int> imgBasis = {2*edge, 2*edge +1};
        
        // Vote for the tables using the given basis
        vector<HashTable> votedTables = hashing::voteForTables(tables, imgPoints, imgBasis);
        int maxVotes = votedTables[0].votes;
        cout << "MAX VOTES = " << maxVotes << endl << endl;
        
        for (int i = 0; i < votedTables.size(); i++) {
            HashTable t = votedTables[i];
            if (t.votes < MIN(200, maxVotes)) break;
            
            vector<Mat> orderedPoints = hashing::getOrderedPoints(imgBasis, t, modelPoints3D, imgPoints);
            
            Mat newModel = orderedPoints[0];
            Mat newTarget = orderedPoints[1];
            
            // Require at least 4 correspondences
            if (newModel.cols < 4) {
                cout << "* * * Only " << newModel.cols << " correspondences!!" << endl;
                continue;
            }
            
            vconcat(newModel.rowRange(0, 2), newModel.row(3), newModel);
            vconcat(newTarget.t(), Mat::ones(1, newTarget.rows, newTarget.type()), newTarget);
            
            Mat modInv = newModel.t() * (newModel * newModel.t()).inv();
            H = newTarget * modInv;
            cout << H << endl;
            
        }
        edge++;
    }
    
    auto endRecog = chrono::system_clock::now();
    chrono::duration<double> timeRecog = endRecog-startRecog;
    cout << "Recognition time = " << timeRecog.count()*1000.0 << " ms" << endl;
    
    float theta = atan2(H.at<float>(0,1), H.at<float>(0,0));
    cout << "theta = " << theta << endl;
    
    Point2f translation = Point2f(H.at<float>(0,2), H.at<float>(1,2));
    cout << "translation = " << translation << endl;
    
    waitKey(0);
    
    return 0;
}
