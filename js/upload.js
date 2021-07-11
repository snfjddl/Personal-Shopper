const readURL = (input) => {
    if (input.files && input.files[0]) {
        const reader = new FileReader();
        reader.onload = (e) => {
            $('.image-upload-wrap').hide();
            $('.file-upload-image').attr('src', e.target.result);
            $('.file-upload-content').show();
            $('.image-title').html(input.files[0].name);
        };
        reader.readAsDataURL(input.files[0]);
    } else {
        removeUpload();
    }
}

function removeUpload() {
    $('.file-upload-input').replaceWith($('.file-upload-input').clone());
    $('.file-upload-content').hide();
    $('.image-upload-wrap').show();
}

function uploadFile(){
    if (confirm('파일을 업로드하시겠습니까?')) {
        var data = new FormData();
        data.append("dressImg", $('.file-upload-input').prop('files')[0]);
        console.log(data);

        $.ajax({
            type: "POST",
            enctype: 'multipart/form-data',
            url: "http://141.223.140.19:5000/predict-image",
            data: data,
            processData: false,
            contentType: false,
            cache: false,
            timeout: 600000,
            success: function(result) {
                console.log("SUCCESS : ", result);
                alert("업로드 성공")
            },
            error: function(e) {
                console.log("ERROR : ", e);
            }
        });
    }
}