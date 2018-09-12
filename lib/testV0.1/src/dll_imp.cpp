#include <Windows.h>
//#include "newface.h"
#include"dll_imp.h"

//Ini_Modify_Face init_modify_face;
//Get_Modify_Face get_modify_face;
Ini_Face init_face;
Get_Vec   get_vec;
Cal_Cos  cal_cos;
Set_Face_Ext set_face_ext;
Set_Dis_Threshold set_distance_threshold;

void init_face_imp(const char* dllfile)
{
	HINSTANCE hDllInst = LoadLibrary(dllfile);
	if (!hDllInst) printf("load lib error\n");

	init_face = (Ini_Face)GetProcAddress(hDllInst, "init_face");
	if (!init_face) printf("load init_face error\n");
	//init_modify_face = (Ini_Modify_Face)GetProcAddress(hDllInst, "init_modify_face");
	//get_modify_face = (Get_Modify_Face)GetProcAddress(hDllInst, "get_modify_face");	
	get_vec = (Get_Vec)GetProcAddress(hDllInst, "get_vec");
	cal_cos = (Cal_Cos)GetProcAddress(hDllInst, "cal_cos");
	set_face_ext = (Set_Face_Ext)GetProcAddress(hDllInst, "set_face_ext");
	set_distance_threshold = (Set_Dis_Threshold)GetProcAddress(hDllInst, "set_distance_threshold");
}