#include <Windows.h>
#include"dll_imp.h"

Init_Net   init_net;
Predict   predict;
Set_Input  set_input;
Get_Output get_output;
Get_Dims   get_dims;
Vec2mat    vec2mat;


void init_imp(const char* dllfile)
{
	HINSTANCE hDllInst = LoadLibrary(dllfile);
	if (!hDllInst) printf("load lib error\n");

	init_net = (Init_Net)GetProcAddress(hDllInst, "init_net");
	if (!init_net) printf("load init_net error\n");

	predict = (Predict)GetProcAddress(hDllInst, "predict");
	set_input = (Set_Input)GetProcAddress(hDllInst, "set_input");
	get_output = (Get_Output)GetProcAddress(hDllInst, "get_output");
	get_dims = (Get_Dims)GetProcAddress(hDllInst, "get_dims");
	vec2mat = (Vec2mat)GetProcAddress(hDllInst, "vec2mat");
}