//
//  boundsType.h
//  reac2
//
//  Created by Антон Кудряшов on 17.12.15.
//  Copyright © 2015 Антон Кудряшов. All rights reserved.
//

#ifndef boundsType_h
#define boundsType_h

enum boundsType {
    boundsNormal = 0,
    boundsCyclic = 1
};

struct boundValue { //y_o = k*y_1 + r; y_N = k*y_(N-1) + r;
    boundsType type;
    double k;
    double r;
};

struct boundsConditions {
    boundValue left;
    boundValue right;
};

#endif /* boundsType_h */
