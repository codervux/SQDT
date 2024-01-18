import streamlit as st
import cv2
import os


def main():
    new_title = '<p style="font-size: 42px;">Welcome to my HOI Detection App!</p>'
    read_me_0 = st.markdown(new_title, unsafe_allow_html=True)

    read_me = st.markdown("""
    This project was built using Streamlit and OpenCV 
    to demonstrate SQDT HOI detection in both videos(pre-recorded)
    and images.
    
    
    This SQDT HOI Detection project can detect the triplet [H, O, I]
    in either a video or image. """
    )
    st.sidebar.title("Select Activity")
    choice  = st.sidebar.selectbox("MODE",("About","HOI Detection(Image)","HOI Detection(Video)"))

    if choice == "HOI Detection(Image)":
        #st.subheader("HOI Detection")
        read_me_0.empty()
        read_me.empty()
        #st.title('HOI Detection')
        HOI_detection_image()


    elif choice == "About":
        print() 

def HOI_detection_image():
    st.title('HOI Detection for Images')
    st.subheader("""
    This HOI Detection project takes in an image and outputs the image with bounding boxes created around the objects in the image, and the action class, object class that human interact with""")
    file = st.file_uploader('Upload Image', type = ['jpg','png','jpeg'])

    if file != None:
        st.write("Uploaded test {}".format(file.name))
        inputpath = os.path.join("image",file.name)
        st.image(inputpath)
        outpath = "Visualize/result.png"
        #print(outpath)
        with open(inputpath, "wb") as f:
            f.write(file.getbuffer())

        os.system("python main.py --HOIDet --share_enc --pretrained_dec --num_hoi_queries 16 --object_threshold 0 --temperature 0.2 --no_aux_loss --eval --dataset_file hico-det --resume hico_ft_q16.pth --input_path " + inputpath)
        print("DONE")
        st.image(outpath,"HOI Detection Result")
        os.remove("./Visualize/result.png")
if __name__ == '__main__':
		main()