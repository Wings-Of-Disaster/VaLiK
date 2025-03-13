function getData {
    cd datasets && \
    wget https://crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v2.0.tar.gz && \ 
    tar -zxf CrisisMMD_v2.0.tar.gz && \
    cd CrisisMMD_v2.0 && \ 
    unzip crisismmd_datasplit_all.zip && \
    rm crisismmd_datasplit_all.zip && \
    cd .. && \
    rm CrisisMMD_v2.0.tar.gz && \
    cd ..
}

function ProcessData_CLIP_Interrogator {
    cd src
    cd Image_to_Text
    python CLIP_Interrogator_CrisisMMD.py
    cd ..
    cd Original_Text_Compilation
    python Get_Text_CrisisMMD.py
    cd ..
    cd ..
}

#function ProcessData_BLIP2 {
    #cd src
    #cd Image_to_Text
    #python BLIP2_CrisisMMD.py --input ../../datasets/CrisisMMD_v2.0/data_image --model_type flan-t5
    #cd ..
    #cd Original_Text_Compilation
    #python Get_Text_CrisisMMD.py
    #cd ..
    #cd ..
#}

echo "Checking if CrisisMMD_v2.0 dataset exists..."
[ ! -d "datasets" ] && mkdir datasets
[ ! -d "datasets/CrisisMMD_v2.0" ] && getData

echo "Processing data... Please wait..."
ProcessData_CLIP_Interrogator