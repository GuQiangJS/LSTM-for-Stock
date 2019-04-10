// Basic example
$(document).ready(function () {
$('#dtBasicExample').DataTable({
	"searching":false,
    "pageLength": 15,
	"lengthChange":false
});
$('.dataTables_length').addClass('bs-select');
});
$(".price").each(function(){
    this.innerText=numeral(this.innerText).format('0.0000')
})
$(".percent").each(function(){
    this.innerText=numeral(this.innerText/100).format('0.00%')
})