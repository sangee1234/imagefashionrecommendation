function showImage(input){
  fileupload = document.getElementById("my_file");
  document.getElementById("im1").src="static/img1.png";
  fileupload.click(input);
}

window.onload = function (input) {
var fileupload = document.getElementById("my_file");
fileupload.onclick = function () {
  var fileName = fileupload.value.split('\\')[fileupload.value.split('\\').length - 1];
};

}


async function loadFile(event){
  // let file = document.getElementById("my_file").files[0];
  var fd = new FormData();
  fd.append('image', document.getElementById("my_file").files[0] /*, optional filename */)
  // let formData = new FormData();
  // formData.append('profile-image', document.getElementById("uploadDP").value);
  const response = await fetch("http://localhost:5000/detect", {
  method: 'POST',
  headers: {
    // Content-Type may need to be completely **omitted**
    // or you may need something
  },
  body: fd })
  const data = await response.json()
  displayResults(data);

  var output = document.getElementById('im1');
  output.src = URL.createObjectURL(event.target.files[0]);

  // $('#contents').css("display","none");
  // $('#contents2').css("display","grid");

  // fetch("https://api.publicapis.org/entries")
  // .then((res) => res.json())
  // .then(function (data) {
  //   displayResults(data);
  // })
  // .catch((err) => console.log(err));

function displayResults(data) {
  const cardSelector = document.getElementById("contents2");
  cardSelector.setAttribute("style","display: grid;grid-template-columns: repeat(3, 1fr);");
  data.entries.slice(0, 9).forEach((result) => {
    let card = `
              <div class="card" style="width: 18rem; height: 17rem;">
              <img class="card-img-top" src="static/img3.png" style="width: 17rem; height: 15rem;" alt="Card image cap">
                <div class="card-body">
                  <p class="card-text">Product name.</p>
                  <a href="#" class="btn btn-primary">Visit product</a>
                </div>
              </div>`;

    let containerDiv = document.createElement("div"); //create the col div
    containerDiv.setAttribute("class", "col-sm-8 col-md-3"); //Add the bootstrap col class as needed
    containerDiv.innerHTML = card;
    cardSelector.appendChild(containerDiv);
  });
}
}