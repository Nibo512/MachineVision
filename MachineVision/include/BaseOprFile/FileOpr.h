#pragma once
#include "utils.h"

//��ȡCSV�ļ�
bool ReadCSVFile(const string& filename, vector<vector<string>>& strArray);

//д��CSV�ļ�
bool WriteCSVFile(vector<vector<string>>& strArray, const string &filename);

//��ȡ�궨�ļ�
bool readBinFile(const string& filename, vector<double>& data);
