/*
 * display.h
 *
 *  Created on: 07.10.2012
 *      Author: iwan
 */
#include <stdlib.h>
#include <stdio.h>

#ifndef DISPLAY_H_
#define DISPLAY_H_

void initDisplay(void);
void showGalaxy(float *data, int bodies, bool consoleoutput);
void framerateUpdate(void);
void closeWindow(void);
#endif /* DISPLAY_H_ */
