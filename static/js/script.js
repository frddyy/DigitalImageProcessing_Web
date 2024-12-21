$(document).ready(function() {
    // start: Sidebar
    $('.sidebar-dropdown-menu').slideUp('fast')

    $('.sidebar-menu-item.has-dropdown > a, .sidebar-dropdown-menu-item.has-dropdown > a').click(function(e) {
        e.preventDefault()

        if(!($(this).parent().hasClass('focused'))) {
            $(this).parent().parent().find('.sidebar-dropdown-menu').slideUp('fast')
            $(this).parent().parent().find('.has-dropdown').removeClass('focused')
        }

        $(this).next().slideToggle('fast')
        $(this).parent().toggleClass('focused')
    })

    $('.sidebar-toggle').click(function() {
        $('.sidebar').toggleClass('collapsed')

        $('.sidebar.collapsed').mouseleave(function() {
            $('.sidebar-dropdown-menu').slideUp('fast')
            $('.sidebar-menu-item.has-dropdown, .sidebar-dropdown-menu-item.has-dropdown').removeClass('focused')
        })
    })

    $('.sidebar-overlay').click(function() {
        $('.sidebar').addClass('collapsed')

        $('.sidebar-dropdown-menu').slideUp('fast')
        $('.sidebar-menu-item.has-dropdown, .sidebar-dropdown-menu-item.has-dropdown').removeClass('focused')
    })

    if(window.innerWidth < 768) {
        $('.sidebar').addClass('collapsed')
    }
    // end: Sidebar



    // start: Charts
    const labels = [
        'January',
        'February',
        'March',
        'April',
        'May',
        'June',
    ];

    const salesChart = new Chart($('#sales-chart'), {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                backgroundColor: '#6610f2',
                data: [5, 10, 5, 2, 20, 30, 45],
            }]
        },
        options: {
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    })

    const visitorsChart = new Chart($('#visitors-chart'), {
        type: 'doughnut',
        data: {
            labels: ['Children', 'Teenager', 'Parent'],
            datasets: [{
                backgroundColor: ['#6610f2', '#198754', '#ffc107'],
                data: [40, 60, 80],
            }]
        }
    })
    // end: Charts
})

// Fungsi untuk menampilkan nama gambar pada elemen <p>
function showHint() {
    var gameImages = document.querySelectorAll(".col-md-3");

    gameImages.forEach(function (element) {
        var photoName = element.querySelector("p");
        photoName.classList.remove("visually-hidden");
    });

     // Setelah 3 detik, kembalikan elemen ke keadaan semula
    setTimeout(function () {
        gameImages.forEach(function (element) {
            var photoName = element.querySelector("p");
            photoName.classList.add("visually-hidden");
        });
    }, 10000); // 10000 milidetik (10 detik)
}

// Tambahkan event listener ke tombol "Hint"
document.getElementById("hintBtn").addEventListener("click", showHint);

function reset() {
    confirm("Anda yakin ingin mereset permainan ini?");
    location.reload();
}

document.getElementById("resetBtn").addEventListener("click", reset);

function surrender() {
    var gameImages = document.querySelectorAll(".col-md-3");

    gameImages.forEach(function (element) {
        var imgElement = element.querySelector("img");
        imgElement.src = element.getAttribute("data-new-image");
    });

    // Reset array clickedImages
    clickedImages = [];
}

document.getElementById("surrenderBtn").addEventListener("click", surrender);

var clickedImages = []; // Array untuk menyimpan gambar yang telah diklik
var score = 0; // Skor awal

document.querySelector(".row").addEventListener("click", function (event) {
    var clickedElement = event.target.closest(".game-img");
    if (clickedElement) {
        flipImage(clickedElement);
    }
});

function flipImage(element) {
    var imageId = element.querySelector("img").id;
    var newImage = element.getAttribute("data-new-image");
    var originalImage = element.getAttribute("data-ori");

    clickedImages.push(imageId); // Tambahkan ID gambar ke dalam array
    // Tukar gambar
    if (newImage) {
        element.querySelector("img").src = newImage;

        // Tambahkan class 'flipped' untuk memicu animasi
        element.querySelector("img").classList.add("flipped");

        // Setelah 1 detik, hapus class 'flipped'
        setTimeout(function () {
            element.querySelector("img").classList.remove("flipped");

            // Cek apakah ada dua gambar yang memiliki ID yang sama
            if (clickedImages.length === 2) {
                if (clickedImages[0] === clickedImages[1]) {
                    // Jika ID kedua gambar sama, tambahkan skor
                    score += 1;
                    document.getElementById("score").textContent = score;
                    console.log(clickedImages[0] + " " + clickedImages[1] + " Berhasil")

                    // Reset array clickedImages setelah 2 gambar telah diklik
                    clickedImages = [];
                } else {
                    // Jika ID kedua gambar berbeda, kembalikan gambar ke aslinya
                    setTimeout(function () {
                        var prevElement1 = document.querySelector("[id='" + clickedImages[0] + "']").parentElement;
                        var prevElement2 = document.querySelector("[id='" + clickedImages[1] + "']").parentElement;
                        prevElement1.querySelector("img").src = prevElement1.getAttribute("data-ori");
                        prevElement2.querySelector("img").src = prevElement2.getAttribute("data-ori");
                        console.log(clickedImages[0] + " " + clickedImages[1] + " Gagal")
                        clickedImages = [];
                    }, 500);
                }
            }
        }, 500);
    } else {
        // Simpan gambar asli dalam atribut data-new-image
        element.setAttribute("data-new-image", element.querySelector("img").src);
    }
}


