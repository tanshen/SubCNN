/*
 *  Scores = compute_matching_scores_segmentation(Boxes_det, Patterns_det, Boxes_gt, Patterns_gt)
 *
 *  Inputs:
 *
 *  Detections: N X 4 double matrix where D(i,:) = [x1 y1 x2 y2]
 *  Patterns: height x width x N unit8 matrix for occlusion patterns
 *
 *  Outputs:
 *
 *  Scores: N by M matching score matrix between detections and groundtruth
 */

#include "mex.h"

#define MIN(A,B) ((A) > (B) ? (B) : (A))
#define MAX(A,B) ((A) < (B) ? (B) : (A))

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  size_t num_det, num_gt, height, width;
  const mwSize *dims_det;
  double *scores, *detections, *gts;
  unsigned char *patterns, *patterns_gt;

  num_det = mxGetM(prhs[0]);
  num_gt = mxGetM(prhs[2]);
  if(num_det == 0 || num_gt == 0)
  {
    plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
    return;
  }

  // Error check
  if (mxIsDouble(prhs[0]) == false) 
    mexErrMsgTxt("Box det is not double");
  if (mxIsClass(prhs[1], "uint8") == false) 
    mexErrMsgTxt("Pattern det is not uint8");

  if (mxIsDouble(prhs[2]) == false) 
    mexErrMsgTxt("Box gt is not double");
  if (mxIsClass(prhs[3], "uint8") == false) 
    mexErrMsgTxt("Pattern gt is not uint8");

  if(mxGetNumberOfDimensions(prhs[0]) != 2)
    mexErrMsgTxt("Box det is not two dimentions");
  if(mxGetNumberOfDimensions(prhs[1]) != 2 && mxGetNumberOfDimensions(prhs[1]) != 3)
    mexErrMsgTxt("Pattern det is not two or three dimentions");

  if(mxGetNumberOfDimensions(prhs[2]) != 2)
    mexErrMsgTxt("Box gt is not two dimentions");
  if(mxGetNumberOfDimensions(prhs[3]) != 2 && mxGetNumberOfDimensions(prhs[3]) != 3)
    mexErrMsgTxt("Pattern gt is not two or three dimentions");

  dims_det = mxGetDimensions(prhs[1]);
  height = dims_det[0];
  width = dims_det[1];

  // input
  detections = mxGetPr(prhs[0]);
  patterns = (unsigned char *)mxGetData(prhs[1]);
  gts = mxGetPr(prhs[2]);
  patterns_gt = (unsigned char *)mxGetData(prhs[3]);

  // output
  plhs[0] = mxCreateDoubleMatrix(num_det, num_gt, mxREAL);
  scores = mxGetPr(plhs[0]);

  // compute the area for each detection pattern
  double *x1 = detections + 0*num_det;
  double *y1 = detections + 1*num_det;
  double *x2 = detections + 2*num_det;
  double *y2 = detections + 3*num_det;
  double *area = (double*)mxCalloc(num_det, sizeof(double));
  for(int i = 0; i < num_det; i++)
  {
    unsigned char *p = patterns + i*height*width;
    int count = 0;
    for(int x = x1[i]-1; x < x2[i]; x++)
    {
      for(int y = y1[i]-1; y < y2[i]; y++)
      {
        if(p[x*height + y] > 0)
          count++;
      }
    }
    area[i] = count;
  }

  // compute the area for each gt
  double *x1_gt = gts + 0*num_gt;
  double *y1_gt = gts + 1*num_gt;
  double *x2_gt = gts + 2*num_gt;
  double *y2_gt = gts + 3*num_gt;
  double *area_gt = (double*)mxCalloc(num_gt, sizeof(double));
  for(int i = 0; i < num_gt; i++)
  {
    unsigned char *p = patterns_gt + i*height*width;
    int count = 0;
    if(x1_gt[i] == 0)
    {
      area_gt[i] = 0;
      continue;
    }

    for(int x = x1_gt[i]-1; x < x2_gt[i]; x++)
    {
      for(int y = y1_gt[i]-1; y < y2_gt[i]; y++)
      {
        if(p[x*height + y] > 0)
          count++;
      }
    }
    area_gt[i] = count;
  }

  // compute the matching scores
  for(int i = 0; i < num_det; i++)
  {
    for(int j = 0; j < num_gt; j++)
    {
      double score = 0;

      // compute bounding box overlap
  	  double w = MIN(x2[i], x2_gt[j]) - MAX(x1[i], x1_gt[j]) + 1;
	    double h = MIN(y2[i], y2_gt[j]) - MAX(y1[i], y1_gt[j]) + 1;
      if (w > 0 && h > 0)
      {
        unsigned char *pi = patterns + i*height*width;
        unsigned char *pj = patterns_gt + j*height*width;

        // compute the overlap
        double overlap = 0;
        int xmin = MIN(x1[i], x1_gt[j]);
        int xmax = MAX(x2[i], x2_gt[j]);
        int ymin = MIN(y1[i], y1_gt[j]);
        int ymax = MAX(y2[i], y2_gt[j]);
        for(int x = xmin-1; x < xmax; x++)
        {
          for(int y = ymin-1; y < ymax; y++)
          {
            if(pi[x*height+y] > 0 && pj[x*height+y] > 0)
              overlap++;
          }
        }

        score = overlap / (area[i] + area_gt[j] - overlap);
      }

      // assign the value
      scores[j*num_det + i] = score;
    }
  }

  mxFree(area); 
  mxFree(area_gt); 
}
