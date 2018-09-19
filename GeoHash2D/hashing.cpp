//
//  hashing.cpp
//  GeoHash
//
//  Created by Daniel Mesham on 15/08/2018.
//  Copyright © 2018 Daniel Mesham. All rights reserved.
//

#include "hashing.hpp"


HashTable hashing::createTable(vector<int> basisID, vector<Point2f> modelPoints, vector<bool> visible, float binWidth) {
    // Creates a hash table using the given point indices to determine the basis points. All remaining model points are then hashed acording to their basis-relative coordinates
    geo_hash gh(binWidth);
    vector<Point2f> basis = { modelPoints[basisID[0]], modelPoints[basisID[1]] };
    vector<bin_index> basisBins;
    
    // Create a list of the bins that correspond to basis points
    for (int i = 0; i < basis.size(); i++) {
        Point2f bc = basisCoords(basis, basis[i]);
        basisBins.push_back(gh.point_to_bin(point(bc.x, bc.y)));
    }
    
    for (int j = 0; j < modelPoints.size(); j++) {
        // Do not hash basis or invisible points
        if (j == basisID[0] || j == basisID[1] || !visible[j]) continue;
        Point2f bc = basisCoords(basis, modelPoints[j]);
        point pt = point(bc.x, bc.y, j);
        // Don't add if in a basis bin
        bool inBin = false;
        for (int b = 0; b < basisBins.size(); b++) {
            if (gh.point_to_bin(pt).equals(basisBins[b])) {
                inBin = true;
                break;
            }
        }
        if (!inBin) gh.add_point( pt );
    }
    
    return HashTable(gh, basisID);
}


vector<HashTable> hashing::voteForTables(vector<HashTable> tables, vector<Point2f> imgPoints, vector<int> imgBasis) {
    // Generates a vote for each table based on how many model points lie in the same bin as a given image point when in the coordinate system of the given basis.
    // Sorts the table based on the number of votes.
    tables = clearVotes(tables);
    
    vector<Point2f> basis = { imgPoints[imgBasis[0]], imgPoints[imgBasis[1]] };
    
    for (int i = 0; i < imgPoints.size(); i++) {
        if (i == imgBasis[0] || i == imgBasis[1]) continue;
        Point2f bc = basisCoords(basis, imgPoints[i]);
        point pt = point(bc.x, bc.y);
        
        // Check for matches in each table
        for (int j = 0; j < tables.size(); j++) {
            vector<point> points = tables[j].table.points_in_bin(pt);
            tables[j].votes += points.size();
            //if (points.size() > 0) tables[j].votes += 1;
        }
    }
    sort(tables.begin(), tables.end(), greater<HashTable>());
    return tables;
}

vector<HashTable> hashing::clearVotes(vector<HashTable> tables) {
    for (int i = 0; i < tables.size(); i++) {
        tables[i].votes = 0;
    }
    return tables;
}


vector<Mat> hashing::getOrderedPoints(vector<int> imgBasis, HashTable ht, vector<Point3f> modelPoints, vector<Point2f> imgPoints) {
    // Returns a Mat of model points and image points for use in the least squares algorithm. The orders of both are the same (i.e. the i-th model point corresponds to the i-th image point).
    vector<int> basisIndex = ht.basis;
    vector<Point3f> orderedModelPoints;
    vector<Point2f> orderedImgPoints;
    vector<Point2f> basis = { imgPoints[imgBasis[0]], imgPoints[imgBasis[1]] };
    
    for (int j = 0; j < imgPoints.size(); j++) {
        Point2f bc = basisCoords(basis, imgPoints[j]);
        
        // If a basis point...
        if (j == imgBasis[0]) {
            orderedModelPoints.push_back(modelPoints[basisIndex[0]]);
            orderedImgPoints.push_back(imgPoints[j]);
        }
        else if (j == imgBasis[1]) {
            orderedModelPoints.push_back(modelPoints[basisIndex[1]]);
            orderedImgPoints.push_back(imgPoints[j]);
        }
        
        // If not a basis point...
        else {
            point pt = point(bc.x, bc.y);
            vector<point> binPoints = ht.table.points_in_bin(pt);
            
            if (binPoints.size() > 0) {
                // Take the first point in the bin
                int modelPt_ID = binPoints[0].getID();
                
                orderedModelPoints.push_back(modelPoints[modelPt_ID]);
                orderedImgPoints.push_back(imgPoints[j]);
            }
        }
    }
    
    Mat newModel = pointsToMat3D_Homog(orderedModelPoints);
    Mat imgTarget = pointsToMat2D(orderedImgPoints).t();
    return {newModel, imgTarget};
}


Point2f hashing::basisCoords(vector<Point2f> basis, Point2f p) {
    // Converts the coordinates of point p into the reference frame with the given basis
    Point2f O = (basis[0] + basis[1])/2;
    basis[0] -= O;
    basis[1] -= O;
    p = p - O;
    
    float B = sqrt(pow(basis[1].x, 2) + pow(basis[1].y, 2));
    float co = basis[1].x / B;
    float si = basis[1].y / B;
    
    float u =  co * p.x + si * p.y;
    float v = -si * p.x + co * p.y;
    
    return Point2f(u, v)/B;
}

Mat hashing::pointsToMat2D(vector<Point2f> points) {
    int rows = 2;
    int cols = int(points.size());
    
    float table[rows][cols];
    for (int c = 0; c < cols; c++) {
        table[0][c] = points[c].x;
        table[1][c] = points[c].y;
    }
    return Mat(rows, cols, CV_32FC1, table) * 1;
}

Mat hashing::pointsToMat3D(vector<Point3f> points) {
    int rows = 3;
    int cols = int(points.size());
    
    float table[rows][cols];
    for (int c = 0; c < cols; c++) {
        table[0][c] = points[c].x;
        table[1][c] = points[c].y;
        table[2][c] = points[c].z;
    }
    return Mat(rows, cols, CV_32FC1, table) * 1;
}

Mat hashing::pointsToMat3D_Homog(vector<Point3f> modelPoints) {
    // Converts 3D model points into their full 3D homogeneous representation
    Mat m = pointsToMat3D(modelPoints);
    
    Mat one = Mat::ones(1, m.cols, CV_32FC1);
    vconcat(m, one, m);
    
    return m * 1;
}
