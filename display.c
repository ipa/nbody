#include<X11/Xlib.h>
#include<X11/Xutil.h>

#include<stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#include "display.h"

#define BORDER_WIDTH 2

/* Program wide globals */
Display *theDisplay;
Window theWindow;
GC theGC;
int theScreen;
int theDepth;
unsigned long theBlackPixel;
unsigned long theWhitePixel;

void initX(void) {
}

int createGC(Window theNewWindow, GC *theNewGC) {
	XGCValues theGCValues;

	*theNewGC = XCreateGC(theDisplay, theNewWindow, (unsigned long) 0,
			&theGCValues);
	if (*theNewGC == 0)
		return 0;
	else {
		XSetForeground(theDisplay, *theNewGC, theBlackPixel);
		XSetBackground(theDisplay, *theNewGC, theWhitePixel);
		return 1;
	}
}

Window OpenWindow(int x, int y, int width, int height, int flag, GC *theNewGC) {
	XSetWindowAttributes theWindowAttributes;
	unsigned long theWindowMask;
	XSizeHints theSizeHints;
	XWMHints theWMHints;
	Window theNewWindow;

	/*Setting the attributes*/
	theWindowAttributes.border_pixel = BlackPixel(theDisplay,theScreen) ;
	theWindowAttributes.background_pixel = WhitePixel(theDisplay,theScreen) ;
	theWindowAttributes.override_redirect = False;

	theWindowMask = CWBackPixel | CWBorderPixel | CWOverrideRedirect;

	theNewWindow = XCreateWindow(theDisplay, RootWindow(theDisplay,theScreen) ,
	x,y,width,height,
	BORDER_WIDTH,theDepth,
	InputOutput,
	CopyFromParent,
	theWindowMask,
	&theWindowAttributes);

	theWMHints.initial_state = NormalState;
	theWMHints.flags = StateHint;

	XSetWMHints(theDisplay, theNewWindow, &theWMHints);

	theSizeHints.flags = PPosition | PSize;
	theSizeHints.x = x;
	theSizeHints.y = y;
	theSizeHints.width = width;
	theSizeHints.height = height;

	XSetNormalHints(theDisplay, theNewWindow, &theSizeHints);

	if (createGC(theNewWindow, theNewGC) == 0) {
		XDestroyWindow(theDisplay, theNewWindow);
		return ((Window) 0);
	}

	XMapWindow(theDisplay, theNewWindow);
	XFlush(theDisplay);

	return theNewWindow;
}

void initDisplay() {
	theDisplay = XOpenDisplay(NULL );

	theScreen = DefaultScreen(theDisplay);
	theDepth = DefaultDepth(theDisplay,theScreen) ;
	theBlackPixel = BlackPixel(theDisplay,theScreen) ;
	theWhitePixel = WhitePixel(theDisplay,theScreen) ;
	theWindow = OpenWindow(50, 50, 800, 800, 0, &theGC);

	XDrawString(theDisplay, theWindow, theGC, 10, 10, "NBody", strlen("NBody"));

	XFlush(theDisplay);

}

void clear(){
	XClearWindow(theDisplay, theWindow);
}

void showGalaxy(float* data, int bodies, bool consoleoutput) {
	int i, idx, vidx;
	clear();
	for (i = 0; i < bodies; i++) {
		// float array index
		idx = i * 4;
		vidx = bodies * 4 + idx;

		if(consoleoutput){
			printf("g: %d gpos: %f %f %f %f, gvel: %f %f %f %f \n", i,
				data[idx + 0], data[idx + 1], data[idx + 2], data[idx + 3],
				data[vidx + 0], data[vidx + 1], data[vidx + 2], data[vidx + 3]);
		}
		int size = (int)floor(data[idx + 3] / 10.0);
		int x = data[idx + 0] + 150;
		int y = data[idx + 1] + 150;
		x -= size/2;
		y -= size/2;
		XFillArc(theDisplay, theWindow, theGC, x, y, size, size, 0,23040);
	}

	framerateUpdate();
	XFlush(theDisplay);
}

void closeWindow(){
	XCloseDisplay(theDisplay);
}

#include <sys/time.h>
#include <stdio.h>

struct timeval frameStartTime, frameEndTime;
char	appTitle[64] = "NBody Simulation";			// window title
float	refreshtime = 1.0f;						// fps refresh period
float	gElapsedTime = 0.0f;					// current frame elapsed time
float	gTimeAccum = 0.0f;						// time accumulator for refresh
int		gFrames = 0;							// frame accumulator
float	gFPS = 0.0f;							// framerate

void framerateTitle(char* title) {
	strcpy(appTitle, title);
}

void framerateUpdate(void)
{
	gettimeofday(&frameEndTime, NULL);

	gElapsedTime = frameEndTime.tv_sec - frameStartTime.tv_sec +
             ((frameEndTime.tv_usec - frameStartTime.tv_usec)/1.0E6);
    frameStartTime = frameEndTime;

    gTimeAccum += gElapsedTime;
    gFrames++;

	if (gTimeAccum > refreshtime)
	{
		char title[64];
		gFPS = (float) gFrames / gTimeAccum;
		sprintf(title, "%s : %3.1f fps", appTitle, gFPS);

		XStoreName(theDisplay, theWindow, title);
		gTimeAccum = 0.0f;
		gFrames = 0;
	}

}
