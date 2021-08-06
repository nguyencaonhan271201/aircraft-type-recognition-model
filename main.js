let form = document.querySelector("#form");
let result_div = document.querySelector(".result");

const proxyUrl = "https://cors-anywhere.herokuapp.com/"

if (form) {
    form.addEventListener("submit", function(e) {
        e.preventDefault();
    
        //Get input
        let SBD = document.querySelector("#SBD").value;
        let fullname = document.querySelector("#name").value;
        var e1 = document.getElementById("combination");
        var combination = e1.options[e1.selectedIndex].value;
    
        let xhr = new XMLHttpRequest();
        data = {
            "SBD": SBD,
            "fullname": fullname,
            "combination": combination
        };
        xhr.open(
            "post",
            "https://nodejstnthpt.herokuapp.com/get_rank/",
            //`${proxyUrl}http://localhost:7890/get_rank?SBD=${SBD}&fullname=${fullname}&combination=${combination}`,
            true
        );
        xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
        //xhr.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");
        xhr.onload = function() {
            if (xhr.status == 200) {
                let result = JSON.parse(xhr.responseText);
                result_div.innerHTML = `
                    <p><b>${result['Họ và tên']} - 0${result['SBD']}</b></p>
                    <p>Điểm khối ${combination} của bạn: <b>${result[combination]}</b></p>
                    <p>Bạn xếp hạng thứ <b>${parseInt(result[combination + "_rank"])}</b> trong toàn bộ 86262 thí sinh cùng cụm thi.</p>
                    <p>Có <b>${result["same_rank"]}</b> thí sinh có cùng mức điểm với bạn.</p>
                `;
                document.querySelector("#SBD").value = "";
                document.querySelector("#name").value = "";
                document.getElementById("combination").selectedIndex = 0;
            } else {
                result_div.innerHTML = `
                    <p class="error">Đã có lỗi xảy ra. Vui lòng thử lại!</p>
                `
            }
        }
        xhr.send(JSON.stringify(data));
        result_div.innerHTML = `
            <p class="font-italic">Đang lấy thông tin...</p>
        `
    })
}

function showImageBox() {
    $(".image-box").css("display", "flex");
    setTimeout(function() {
        $(".image-box").css("opacity", 1);
    }, 10);
}

function hideImageBox() {
    $(".image-box").css("opacity", 0);
    setTimeout(function() {
        $(".image-box").css("display", "none");
    }, 300);
}

function loadImage(src) {
    $(".image-box img").attr("src", src);
}

document.addEventListener("DOMContentLoaded", function() {
    document.querySelectorAll("img").forEach(image => {
        image.addEventListener("click", (e) => {
            e.preventDefault();
            let getURL = e.target.src;
            loadImage(getURL);
            showImageBox();
        })
    })    

    let imgbox = document.querySelector(".image-box");
    if (imgbox) {
        imgbox.addEventListener("click", function() {
            hideImageBox();
        })
    }
})