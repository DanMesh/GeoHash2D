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
#include <fstream>


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

static float binWidth = 0.2;

static string dataFolder = "../../../../../data/";
static string logFolder = "../../../../../logs/";

static bool DEBUGGING = true;
static bool LOGGING = false;
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
    Model * modelDiamond = new Diamond(Scalar(13, 134, 161));
    
    vector<Model *> model = {modelArrow};
    Model * calibModel = model[0];
    
    // * * * * * * * * * * * * * * * * *
    //   HASHING
    // * * * * * * * * * * * * * * * * *
    
    auto startHash = chrono::system_clock::now(); // Start hashing timer
    
    geo_hash onlyTable = geo_hash(binWidth);
    onlyTable = hashing::hashModelsIntoTable(onlyTable, model);
    
    auto endHash = chrono::system_clock::now();
    chrono::duration<double> timeHash = endHash-startHash;
    cout << "Hashing time = " << timeHash.count()*1000.0 << " ms" << endl;
    
    // * * * * * * * * * * * * * * * * *
    //   OPEN THE FIRST FRAME
    // * * * * * * * * * * * * * * * * *
    
    Mat frame;
    String filename = "C_Arrow_1";
    VideoCapture cap(dataFolder + filename + ".avi");
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
    //   SET UP LOGGER
    // * * * * * * * * * * * * * * * * *
    ofstream log;
    if (LOGGING) {
        log.open(logFolder + filename + "_GH.csv");
        log << "Time";
        for (int m = 0; m < model.size(); m++) log << ";error_Area_ " << m << ";error_LSQ_ " << m << ";tX_ " << m << ";tY_ " << m << ";tZ_ " << m << ";rX_ " << m << ";rY_ " << m << ";rZ_ " << m;
        log << endl;
    }
    
    // * * * * * * * * * * * * * * * * *
    //   INPUT LOOP
    // * * * * * * * * * * * * * * * * *
    
    vector<double> times = {};
    double longestTime = 0.0;
    vector<vector<double>> errorArea = vector<vector<double>>(model.size());
    vector<double> worstError = vector<double>(model.size());
    vector<int> failures(model.size());
    
    while (!frame.empty()) {
        Mat frameOrig;
        frame.copyTo(frameOrig);
        
        // * * * * * * * * * * * * * *
        //      RECOGNITION
        // * * * * * * * * * * * * * *
        
        auto startRecog = chrono::system_clock::now(); // Start recognition timer
                
        // Detect edges
        Mat canny, cannyTest;
        Canny(frame, canny, 40, 120);
        
        // Map edge points to the 2D plane
        // TODO: this could also happen after detecting line segments - try it, might be faster
        
        // Convert the canny image to a list of edge coordinates
        Mat edges;
        findNonZero(canny, edges);
        edges = edges.t();
        Mat edgesRows[2];
        split(edges, edgesRows);
        vconcat(edgesRows[0], edgesRows[1], edges);
        vconcat(edges, Mat::ones(1, edges.cols, edges.type()), edges);
        edges.convertTo(edges, P.type());
        
        // Convert image coords to plane coords
        Mat y = P.inv() * edges;
        Mat z = y.row(2);
        Mat norm;
        vconcat(z, z, norm);
        vconcat(norm, z, norm);
        divide(y, norm, y);
        
        // Draw the 2D plane
        Mat plane = Mat::zeros(canny.rows, canny.cols, canny.type());
        
        for (int i = 0; i < y.cols; i++) {
            Point pt = Point(y.at<float>(0, i) + Ox, y.at<float>(1, i) + Oy);
            circle(plane, pt, 0, Scalar(255));
        }
        
        // Detect line segments
        // TODO: either HoughLinesP or LSD (HLP gives better 'complete' lines)
        vector<Vec4i> lines = orange::borderLines(plane);
        vector<Point2f> imgPoints;
        
        Mat plane2 = Mat::zeros(plane.size(), frame.type());
        //cvtColor(plane, plane2, CV_GRAY2BGR);
        // Convert the edges to a set of points
        for (int i = 0; i < lines.size(); i++) {
            Point p1 = Point(lines[i][0], lines[i][1]);
            Point p2 = Point(lines[i][2], lines[i][3]);
            imgPoints.push_back(Point2f(p1));
            imgPoints.push_back(Point2f(p2));
            
            line(plane2, p1, p2, Scalar(150,150,150));
            circle(plane2, p1, 2, Scalar(0,0,255), -1);
            circle(plane2, p2, 2, Scalar(0,0,255), -1);
        }
        if (DEBUGGING) imshow("lines", plane2);
        
        // * * * * * * * * * * * * * * * *
        //      FIND CORRESPONDENCES
        //        WITH HASH TABLE
        // * * * * * * * * * * * * * * * *
        
        // Check for entries in the hash table
        VoteTally bestVT[model.size()];
        vector<estimate> bestEst(model.size());
        vector<int> bestImgBasis[model.size()];
        
        // Try each line as a potential image basis
        for (int l = 0; l < lines.size(); l++) {
            vector<int> imgBasis = {2*l, 2*l + 1};
            vector<VoteTally> vt = hashing::voteWithBasis(onlyTable, imgPoints, imgBasis);
            
            // Find the best model basis for each model
            vector<bool> gotModel(model.size());    // Whether each model has been found for this image basis
            for (int v = 0; v < vt.size(); v++) {
                int modelNum = vt[v].mb.model;
                if (!gotModel[modelNum]) {
                    gotModel[modelNum] = true;
                    if (bestVT[modelNum].votes <= vt[v].votes) {
                        
                        
                        // Do LSQ to see what the error is
                        vector<Mat> orderedPoints = hashing::getOrderedPoints2(onlyTable, vt[v].mb, imgBasis, model[modelNum]->getVertices(), imgPoints);
                        Mat newModel = orderedPoints[0];
                        Mat newTarget = orderedPoints[1];
                        
                        vconcat(newModel.rowRange(0, 2), newModel.row(3), newModel);
                        vconcat(newTarget.t(), Mat::ones(1, newTarget.rows, newTarget.type()), newTarget);
                        
                        // Use least squares to estimate pose
                        estimate est = lsq::poseEstimate2D({0,0,0}, newModel, newTarget);
                        if (est.error < bestEst[modelNum].error) {
                            bestEst[modelNum] = est;
                            bestVT[modelNum] = vt[v];
                            bestImgBasis[modelNum] = imgBasis;
                        }
                    }
                }
            }
        }
        
        for (int m = 0; m < model.size(); m++) {
            if (bestImgBasis[m].empty()) {
                cout << "Error! Empty image basis!" << endl;
                //continue;
            }
            estimate est = bestEst[m];
            
            // * * * * * * * * * * * * * * * * *
            //      SHOW THE ESTIMATED POSE
            // * * * * * * * * * * * * * * * * *
            
            Mat modelMat = model[m]->pointsToMat();
            vconcat(modelMat.rowRange(0, 2), modelMat.row(3), modelMat);
            
            // Find the coordintes using the LSQ result
            Vec3f poseLSQ = {est.pose[0] - Ox, est.pose[1] - Oy, est.pose[5]};
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
            
        }
        
        auto endRecog = chrono::system_clock::now();
        chrono::duration<double> timeRecog = endRecog-startRecog;
        double time = timeRecog.count()*1000.0;
        cout << "Recognition time = " << time << " ms" << endl;
        
        times.push_back(time);
        if (time > longestTime) longestTime = time;
        
        // Measure and report the area errors
        if (REPORT_ERRORS) {
            for (int m = 0; m < model.size(); m++) {
                estimate est = bestEst[m];
                
                Mat modelMat = model[m]->pointsToMat();
                vconcat(modelMat.rowRange(0, 2), modelMat.row(3), modelMat);
                
                // Find the coordintes using the LSQ result
                Vec3f poseLSQ = {est.pose[0] - Ox, est.pose[1] - Oy, est.pose[5]};
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
                
                Mat seg = orange::segmentByColour(frameOrig, model[m]->colour);
                cvtColor(seg, seg, CV_BGR2GRAY);
                Mat silhouette = Mat(frame.rows, frame.cols,  CV_8UC1, Scalar(0));
                fillPoly(silhouette, &pts, &npts, 1, Scalar(255));
                double areaError = area::areaError(silhouette, seg);
                errorArea[m].push_back(areaError);
                if (areaError > worstError[m]) worstError[m] = areaError;
                if (areaError > 50) failures[m]++;
            }
        }
        
        // Log time and errors
        if (LOGGING) {
            log << time;
            for (int m = 0; m < model.size(); m++) {
                log << ";" << errorArea[m].back() << ";" << bestEst[m].error;
                bestEst[m].pose[2] = lines.size(); // Log number of lines in z coord
                for (int i = 0; i < 6; i++) log << ";" << bestEst[m].pose[i];
            }
            log << endl;
        }
        
        imshow("Frame", frame);
        
        // Get next frame
        cap >> frame;
        
        if (waitKey(1) == 'q') break;
    }
    
    vector<double> meanTime, stdDevTime;
    meanStdDev(times, meanTime, stdDevTime);
    cout << endl << "NEW METHOD:" << endl;
    cout << "Avg time     = " << meanTime[0] << " ms     " << 1000.0/meanTime[0] << " fps" << endl;
    cout << "stdDev time  = " << stdDevTime[0] << " ms" << endl;
    cout << "Longest time = " << longestTime << " ms     " << 1000.0/longestTime << " fps" << endl;
    
    // Report errors
    if (!REPORT_ERRORS) return 0;
    cout << endl << "AREA ERRORS:" << endl << "Model   Mean     StDev    Worst     Failures" << endl;
    for (int m = 0; m < model.size(); m++) {
        vector<double> meanError, stdDevError;
        meanStdDev(errorArea[m], meanError, stdDevError);
        printf("%4i    %5.2f    %5.2f    %6.2f   %4i \n", m, meanError[0], stdDevError[0], worstError[m], failures[m]);
    }
    
    return 0;
}
