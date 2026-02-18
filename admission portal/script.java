document.querySelectorAll(".branch").forEach(branch => {
    const total = parseInt(branch.querySelector(".total").textContent);
    const occupiedInput = branch.querySelector(".occupied");
    const availableSpan = branch.querySelector(".available");

    function updateSeats() {
        let occupied = parseInt(occupiedInput.value) || 0;

        if (occupied > total) {
            alert("Occupied seats cannot exceed total seats!");
            occupiedInput.value = total;
            occupied = total;
        }

        availableSpan.textContent = total - occupied;
    }

    occupiedInput.addEventListener("input", updateSeats);
});
